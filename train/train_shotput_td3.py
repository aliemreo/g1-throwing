"""Train a genuine TD3 shot-put policy while the G1 walks forward.

Unlike ``train_walking_td3.py``, this route has no calibrated throwing
trajectory. A zero-output actor holds the right arm at its neutral pose. TD3
must learn the preparation, drive, release timing, and (in later curriculum
stages) whole-body balance residuals.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from shotput_throw_env import (
    SHOTPUT_ACTION_GROUPS,
    SHOTPUT_N_ACTIONS,
    ShotPutWalkingEnv,
)
from shotput_throw_gym import make_shotput_env
from walk_policy import DEFAULT_POLICY


def clean_number(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def default_name(target, command):
    return (
        f"td3_shotput_walk_{clean_number(target[0])}_"
        f"{clean_number(target[1])}_vx{clean_number(command[0])}"
    )


def zero_initialise_actor(model):
    """Make the untrained baseline exactly neutral and reproducible."""
    output = next(
        layer for layer in reversed(list(model.actor.mu.modules()))
        if isinstance(layer, torch.nn.Linear)
    )
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    model.actor_target.load_state_dict(model.actor.state_dict())


def actor_vector(model):
    return torch.cat([
        parameter.detach().cpu().reshape(-1)
        for parameter in model.actor.parameters()
    ]).numpy()


def actor_delta(model, reference):
    current = actor_vector(model)
    if current.shape != reference.shape:
        raise ValueError("actor parameter shapes do not match")
    difference = current - reference
    return {
        "actor_l2_delta": float(np.linalg.norm(difference)),
        "actor_rms_delta": float(np.sqrt(np.mean(
            difference * difference))),
        "actor_parameter_count": int(current.size),
    }


def selection_score(info):
    miss = float(info["objective_miss"])
    return float(
        miss
        + 10.0 * float(info["fallen"])
        + 4.0 * float(info["landing_miss"] is None)
        + 3.0 * (
            1.0 - float(info["shotput_style_score"]))
        + 2.0 * (
            1.0 - float(info["shotput_style_valid"]))
        + max(0.0, 2.0 - float(info["distance_walked"]))
        + 3.0 * max(
            0.0,
            0.45 - float(info["distance_during_throw"]),
        )
        + 2.0 * max(
            0.0,
            float(info["max_abs_lateral_error"]) - 0.22,
        )
    )


def aggregate_infos(infos):
    def mean(key):
        return float(np.mean([
            float(info[key]) for info in infos
        ]))

    result = {
        "selection_score": float(np.mean([
            selection_score(info) for info in infos
        ])),
        "mean_miss": mean("objective_miss"),
        "hit_rate_0_50m": float(np.mean([
            float(info["objective_miss"] < 0.50)
            for info in infos
        ])),
        "fall_rate": mean("fallen"),
        "release_rate": mean("released"),
        "landed_rate": float(np.mean([
            float(info["landing_miss"] is not None)
            for info in infos
        ])),
        "style_valid_rate": mean("shotput_style_valid"),
        "mean_style_score": mean("shotput_style_score"),
        "mean_ready_error": mean("shotput_ready_error"),
        "mean_release_forward_speed": mean(
            "release_forward_speed"),
        "mean_release_hand_height": mean(
            "release_hand_height"),
        "mean_release_elbow_angle": mean(
            "release_elbow_angle"),
        "mean_distance_walked": mean("distance_walked"),
        "mean_distance_during_throw": mean(
            "distance_during_throw"),
        "mean_positive_velocity_fraction_during_throw": mean(
            "positive_velocity_fraction_during_throw"),
        "mean_max_abs_lateral_error": mean(
            "max_abs_lateral_error"),
    }
    return result


def evaluate_model(
    model,
    env_kwargs,
    normalizer,
    episodes=5,
    seed_offset=10_000,
):
    infos = []
    eval_env = ShotPutWalkingEnv(**env_kwargs)
    for episode in range(int(episodes)):
        observation, _ = eval_env.reset(
            seed=seed_offset + episode)
        while True:
            model_observation = observation[None].astype(np.float32)
            if normalizer is not None:
                model_observation = normalizer.normalize_obs(
                    model_observation)
            action, _ = model.predict(
                model_observation,
                deterministic=True,
            )
            observation, _, terminated, truncated, info = (
                eval_env.step(action[0])
            )
            if terminated or truncated:
                infos.append(info)
                break
    return aggregate_infos(infos)


class ShotPutLearningCallback(BaseCallback):
    """Save only deterministic policies that pass learned-style gates."""

    def __init__(
        self,
        out_dir,
        env_kwargs,
        initial_actor,
        initial_score,
        eval_freq=10_000,
        eval_episodes=5,
        min_best_steps=20_000,
        min_improvement=0.05,
        min_style_score=0.35,
        min_style_valid_rate=0.40,
        min_landed_rate=0.60,
        max_fall_rate=0.20,
        window=50,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.env_kwargs = dict(env_kwargs)
        self.initial_actor = initial_actor.copy()
        self.best_score = float(initial_score)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.min_best_steps = int(min_best_steps)
        self.min_improvement = float(min_improvement)
        self.min_style_score = float(min_style_score)
        self.min_style_valid_rate = float(min_style_valid_rate)
        self.min_landed_rate = float(min_landed_rate)
        self.max_fall_rate = float(max_fall_rate)
        self.next_eval = self.eval_freq
        self.misses = deque(maxlen=int(window))
        self.styles = deque(maxlen=int(window))
        self.style_valid = deque(maxlen=int(window))
        self.falls = deque(maxlen=int(window))

        metrics_path = self.out_dir / "best_metrics.json"
        if metrics_path.exists():
            stored = json.loads(
                metrics_path.read_text(encoding="utf-8"))
            self.best_score = min(
                self.best_score,
                float(stored["selection_score"]),
            )

    def _record_result(self, prefix, result):
        for key, value in result.items():
            if isinstance(value, (int, float)):
                self.logger.record(f"{prefix}/{key}", value)

    def _eligible(self, result):
        return bool(
            self.num_timesteps >= self.min_best_steps
            and result["selection_score"]
            < self.best_score - self.min_improvement
            and result["mean_style_score"]
            >= self.min_style_score
            and result["style_valid_rate"]
            >= self.min_style_valid_rate
            and result["landed_rate"]
            >= self.min_landed_rate
            and result["fall_rate"]
            <= self.max_fall_rate
        )

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode_objective_miss" not in info:
                continue
            self.misses.append(float(
                info["episode_objective_miss"]))
            self.styles.append(float(
                info["episode_style_score"]))
            self.style_valid.append(float(
                info["episode_style_valid"]))
            self.falls.append(float(
                info["episode_fallen"]))

        while self.num_timesteps >= self.next_eval:
            normalizer = self.model.get_vec_normalize_env()
            result = evaluate_model(
                self.model,
                self.env_kwargs,
                normalizer,
                episodes=self.eval_episodes,
            )
            result.update(actor_delta(
                self.model,
                self.initial_actor,
            ))
            result["timesteps"] = int(self.num_timesteps)
            self._record_result("shotput_eval", result)
            eligible = self._eligible(result)
            self.logger.record(
                "shotput_eval/eligible_for_best",
                float(eligible),
            )
            self.next_eval += self.eval_freq

            if eligible:
                self.best_score = result["selection_score"]
                self.model.save(str(
                    self.out_dir / "best_model"))
                if normalizer is not None:
                    normalizer.save(str(
                        self.out_dir / "best_vecnorm.pkl"))
                result["source"] = (
                    "TD3 deterministic evaluation after "
                    "gradient updates")
                result["nominal_throw_trajectory"] = False
                result["passes_style_gate"] = True
                (self.out_dir / "best_metrics.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )

        if self.misses:
            self.logger.record(
                "shotput/mean_miss",
                float(np.mean(self.misses)),
            )
            self.logger.record(
                "shotput/mean_style_score",
                float(np.mean(self.styles)),
            )
            self.logger.record(
                "shotput/style_valid_rate",
                float(np.mean(self.style_valid)),
            )
            self.logger.record(
                "shotput/fall_rate",
                float(np.mean(self.falls)),
            )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        default=[4.0, 0.0],
        metavar=("X", "Y"),
    )
    parser.add_argument("--random-target", action="store_true")
    parser.add_argument(
        "--target-r",
        nargs=2,
        type=float,
        default=[3.5, 4.5],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--target-theta",
        nargs=2,
        type=float,
        default=[-0.10, 0.10],
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--command",
        nargs=3,
        type=float,
        default=[1.3, 0.0, 0.0],
        metavar=("VX", "VY", "YAW_RATE"),
    )
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--walk-policy", default=str(DEFAULT_POLICY))
    parser.add_argument(
        "--walk-policy-device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--stage",
        choices=["arm", "upper", "full"],
        default="arm",
        help=(
            "arm: learned right arm/release; upper: also waist and "
            "left arm; full: also learned leg balance residuals"
        ),
    )
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--min-best-steps", type=int, default=20_000)
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--min-style-score", type=float, default=0.35)
    parser.add_argument(
        "--min-style-valid-rate",
        type=float,
        default=0.40,
    )
    parser.add_argument("--min-landed-rate", type=float, default=0.60)
    parser.add_argument("--max-fall-rate", type=float, default=0.20)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--vec-env",
        choices=["subproc", "dummy"],
        default="subproc",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--throw-start", type=float, default=1.4)
    parser.add_argument("--prep-duration", type=float, default=0.70)
    parser.add_argument(
        "--release-delay",
        nargs=2,
        type=float,
        default=[0.12, 0.55],
        metavar=("MIN", "MAX"),
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see an NVIDIA GPU.")
    torch.set_float32_matmul_precision("high")

    target = [float(value) for value in args.target]
    command = [float(value) for value in args.command]
    name = args.name or default_name(target, command)
    out = Path(__file__).resolve().parent / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    actual_device = (
        torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else args.device
    )
    project_root = Path(__file__).resolve().parent.parent
    resolved_walk_policy = Path(args.walk_policy).resolve()
    try:
        stored_walk_policy = str(
            resolved_walk_policy.relative_to(project_root))
    except ValueError:
        stored_walk_policy = str(resolved_walk_policy)

    env_kwargs = {
        "fixed_target": None if args.random_target else target,
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "command": command,
        "throw_start": args.throw_start,
        "prep_duration": args.prep_duration,
        "release_delay": args.release_delay,
        "stage": args.stage,
        "max_seconds": args.max_seconds,
        "walk_policy_path": args.walk_policy,
        "walk_policy_device": args.walk_policy_device,
    }
    config = {
        "algorithm": "TD3",
        "task": (
            "learn a shot-put while continuously walking forward"),
        "controller_design": (
            "pretrained RL walking policy plus learned TD3 shot-put"),
        "nominal_throw_trajectory": False,
        "calibrated_throw_offsets": False,
        "untrained_actor_behavior": (
            "neutral arm; release occurs in the exploration window"),
        "target": target,
        "random_target": args.random_target,
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "command": command,
        "throw_start": args.throw_start,
        "prep_duration": args.prep_duration,
        "release_delay": args.release_delay,
        "stage": args.stage,
        "max_seconds": args.max_seconds,
        "walk_policy": stored_walk_policy,
        "walk_policy_device": args.walk_policy_device,
        "num_envs": args.num_envs,
        "vec_env": args.vec_env,
        "td3_device": args.device,
        "accelerator": actual_device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "action_groups": SHOTPUT_ACTION_GROUPS,
        "action_count": SHOTPUT_N_ACTIONS,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "min_best_steps": args.min_best_steps,
        "min_improvement": args.min_improvement,
        "min_style_score": args.min_style_score,
        "min_style_valid_rate": args.min_style_valid_rate,
        "min_landed_rate": args.min_landed_rate,
        "max_fall_rate": args.max_fall_rate,
        "seed": args.seed,
        "resume": args.resume,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    env_fns = [
        make_shotput_env(
            rank,
            seed=args.seed,
            **env_kwargs,
        )
        for rank in range(args.num_envs)
    ]
    if args.vec_env == "subproc" and args.num_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)

    resume_name = (
        Path(args.resume).stem if args.resume else "")
    normalizer_path = (
        out / "best_vecnorm.pkl"
        if resume_name == "best_model"
        else out / "vecnorm.pkl"
    )
    if args.resume and normalizer_path.exists():
        env = VecNormalize.load(str(normalizer_path), env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    sigma = np.zeros(SHOTPUT_N_ACTIONS, dtype=np.float32)
    sigma[19:23] = [0.22, 0.16, 0.18, 0.22]
    sigma[23] = 0.32
    if args.stage in {"upper", "full"}:
        sigma[12:19] = 0.08
    if args.stage == "full":
        sigma[0:12] = 0.04
    noise = NormalActionNoise(
        np.zeros(SHOTPUT_N_ACTIONS, dtype=np.float32),
        sigma,
    )

    reset_num_timesteps = args.resume is None
    if args.resume:
        model = TD3.load(
            args.resume,
            env=env,
            device=args.device,
            tensorboard_log=str(out / "tb"),
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        for optimizer in (
            model.actor.optimizer,
            model.critic.optimizer,
        ):
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        model.action_noise = noise

        initial_path = out / "initial_model.zip"
        if not initial_path.exists():
            raise FileNotFoundError(
                "initial_model.zip is required to audit learned changes")
        initial_reference_model = TD3.load(
            str(initial_path),
            device="cpu",
        )
        initial_actor = actor_vector(initial_reference_model)
        initial_metrics = json.loads(
            (out / "initial_metrics.json").read_text(
                encoding="utf-8"))
        print(f"resumed from {args.resume}")
    else:
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=noise,
            policy_delay=2,
            target_policy_noise=0.20,
            target_noise_clip=0.40,
            policy_kwargs={"net_arch": [512, 512]},
            tensorboard_log=str(out / "tb"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        zero_initialise_actor(model)
        initial_actor = actor_vector(model)
        initial_metrics = evaluate_model(
            model,
            env_kwargs,
            env,
            episodes=args.eval_episodes,
            seed_offset=20_000,
        )
        initial_metrics.update(actor_delta(
            model,
            initial_actor,
        ))
        initial_metrics.update({
            "timesteps": 0,
            "source": "untrained zero-output TD3 actor",
            "nominal_throw_trajectory": False,
            "is_trained_policy": False,
        })
        model.save(str(out / "initial_model"))
        env.save(str(out / "initial_vecnorm.pkl"))
        (out / "initial_metrics.json").write_text(
            json.dumps(initial_metrics, indent=2),
            encoding="utf-8",
        )
        print(
            "untrained zero actor: "
            f"miss={initial_metrics['mean_miss']:.3f} m, "
            f"style={initial_metrics['mean_style_score']:.3f}, "
            f"valid={initial_metrics['style_valid_rate']:.1%}, "
            f"landed={initial_metrics['landed_rate']:.1%}"
        )

    print(f"TD3 training device: {actual_device}")
    print(
        "Nominal throwing trajectory: NONE "
        "(TD3 must learn the arm motion and release)")
    print(
        f"Curriculum stage: {args.stage}; "
        f"learning starts after {args.learning_starts} samples")

    callback = ShotPutLearningCallback(
        out,
        env_kwargs,
        initial_actor,
        initial_metrics["selection_score"],
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        min_best_steps=args.min_best_steps,
        min_improvement=args.min_improvement,
        min_style_score=args.min_style_score,
        min_style_valid_rate=args.min_style_valid_rate,
        min_landed_rate=args.min_landed_rate,
        max_fall_rate=args.max_fall_rate,
    )
    callbacks = [
        callback,
        CheckpointCallback(
            save_freq=max(
                args.checkpoint_freq // args.num_envs,
                1,
            ),
            save_path=str(out),
            name_prefix="ckpt",
            save_replay_buffer=True,
            save_vecnormalize=True,
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(out / "final"))
        env.save(str(out / "vecnorm.pkl"))
        final_metrics = evaluate_model(
            model,
            env_kwargs,
            env,
            episodes=args.eval_episodes,
            seed_offset=30_000,
        )
        final_metrics.update(actor_delta(
            model,
            initial_actor,
        ))
        final_metrics.update({
            "timesteps_this_invocation": int(args.steps),
            "source": (
                "TD3 deterministic evaluation after training"),
            "nominal_throw_trajectory": False,
            "passes_style_gate": bool(
                final_metrics["mean_style_score"]
                >= args.min_style_score
                and final_metrics["style_valid_rate"]
                >= args.min_style_valid_rate
                and final_metrics["landed_rate"]
                >= args.min_landed_rate
                and final_metrics["fall_rate"]
                <= args.max_fall_rate
            ),
        })
        (out / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2),
            encoding="utf-8",
        )

        proof = {
            "claim": (
                "the shot-put motion is generated by TD3, not a "
                "calibrated throw trajectory"),
            "nominal_throw_trajectory": False,
            "initial": initial_metrics,
            "final": final_metrics,
            "selection_score_improvement": float(
                initial_metrics["selection_score"]
                - final_metrics["selection_score"]
            ),
            "actor_changed": bool(
                final_metrics["actor_l2_delta"] > 1e-8),
            "best_model_exists": bool(
                (out / "best_model.zip").exists()),
            "best_model_rule": (
                "saved only after deterministic improvement and "
                "landing/style gates pass"),
        }
        best_metrics_path = out / "best_metrics.json"
        if best_metrics_path.exists():
            proof["best"] = json.loads(
                best_metrics_path.read_text(encoding="utf-8"))
        (out / "learning_proof.json").write_text(
            json.dumps(proof, indent=2),
            encoding="utf-8",
        )
        env.close()
        print(f"saved training artifacts to {out}")
        print(
            "learning proof: "
            f"{out / 'learning_proof.json'}")


if __name__ == "__main__":
    main()

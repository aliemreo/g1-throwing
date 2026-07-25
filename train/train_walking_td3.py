"""GPU TD3 training for walking forward while throwing a ball."""

import argparse
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from walk_policy import DEFAULT_POLICY
from walking_throw_env import ACTION_GROUPS, N_ACTIONS, WalkingThrowEnv
from walking_throw_gym import make_env


def clean_number(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def default_name(target, command):
    return (
        f"td3_walk_throw_{clean_number(target[0])}_"
        f"{clean_number(target[1])}_vx{clean_number(command[0])}"
    )


def zero_initialise_actor(model):
    """Zero residual initially preserves walking and nominal throwing."""
    output = next(
        layer for layer in reversed(list(model.actor.mu.modules()))
        if isinstance(layer, torch.nn.Linear)
    )
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    model.actor_target.load_state_dict(model.actor.state_dict())


def evaluate_once(model, env):
    observation = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, infos = env.step(action)
        if done[0]:
            return infos[0]


def selection_score(info):
    return float(
        info["objective_miss"]
        + 5.0 * float(info["fallen"])
        + max(0.0, 2.0 - float(info["distance_walked"]))
        + 3.0 * max(
            0.0, 0.4 - float(info["distance_during_throw"]))
        + 2.0 * max(
            0.0,
            0.9 - float(
                info["positive_velocity_fraction_during_throw"]),
        )
        + 2.0 * max(
            0.0, float(info["max_abs_lateral_error"]) - 0.2)
    )


class Level3Callback(BaseCallback):
    """Log actual task metrics and retain the best rolling controller."""

    def __init__(
        self,
        out_dir,
        eval_kwargs,
        window=50,
        initial_best=np.inf,
        eval_freq=5_000,
        eval_episodes=3,
    ):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.eval_kwargs = dict(eval_kwargs)
        self.window = int(window)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.next_eval = self.eval_freq
        self.misses = deque(maxlen=self.window)
        self.falls = deque(maxlen=self.window)
        self.distances = deque(maxlen=self.window)
        self.during_distances = deque(maxlen=self.window)
        self.line_errors = deque(maxlen=self.window)
        self.best_score = float(initial_best)

    def _deterministic_evaluation(self):
        normalizer = self.model.get_vec_normalize_env()
        scores, misses, falls, distances = [], [], [], []
        during_distances, positive_fractions, line_errors = [], [], []
        eval_env = WalkingThrowEnv(**self.eval_kwargs)
        for episode in range(self.eval_episodes):
            observation, _ = eval_env.reset(seed=10_000 + episode)
            while True:
                model_observation = observation[None].astype(np.float32)
                if normalizer is not None:
                    model_observation = normalizer.normalize_obs(
                        model_observation)
                action, _ = self.model.predict(
                    model_observation, deterministic=True)
                observation, _, terminated, truncated, info = (
                    eval_env.step(action[0]))
                if terminated or truncated:
                    scores.append(selection_score(info))
                    misses.append(float(info["objective_miss"]))
                    falls.append(float(info["fallen"]))
                    distances.append(float(info["distance_walked"]))
                    during_distances.append(float(
                        info["distance_during_throw"]))
                    positive_fractions.append(float(
                        info[
                            "positive_velocity_fraction_during_throw"]))
                    line_errors.append(float(
                        info["max_abs_lateral_error"]))
                    break

        result = {
            "selection_score": float(np.mean(scores)),
            "mean_miss": float(np.mean(misses)),
            "fall_rate": float(np.mean(falls)),
            "mean_distance_walked": float(np.mean(distances)),
            "mean_distance_during_throw": float(
                np.mean(during_distances)),
            "mean_positive_velocity_fraction_during_throw": float(
                np.mean(positive_fractions)),
            "mean_max_abs_lateral_error": float(
                np.mean(line_errors)),
        }
        self.logger.record(
            "level3_eval/selection_score", result["selection_score"])
        self.logger.record(
            "level3_eval/mean_miss", result["mean_miss"])
        self.logger.record(
            "level3_eval/fall_rate", result["fall_rate"])
        self.logger.record(
            "level3_eval/mean_distance_walked",
            result["mean_distance_walked"])
        self.logger.record(
            "level3_eval/distance_during_throw",
            result["mean_distance_during_throw"])
        self.logger.record(
            "level3_eval/positive_velocity_fraction_during_throw",
            result[
                "mean_positive_velocity_fraction_during_throw"])
        self.logger.record(
            "level3_eval/max_abs_lateral_error",
            result["mean_max_abs_lateral_error"])
        return result

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode_objective_miss" not in info:
                continue
            self.misses.append(float(info["episode_objective_miss"]))
            self.falls.append(float(info["episode_fallen"]))
            self.distances.append(float(info["episode_distance_walked"]))
            self.during_distances.append(float(
                info["episode_distance_during_throw"]))
            self.line_errors.append(float(
                info["episode_max_abs_lateral_error"]))

        while self.num_timesteps >= self.next_eval:
            result = self._deterministic_evaluation()
            self.next_eval += self.eval_freq
            if result["selection_score"] < self.best_score:
                self.best_score = result["selection_score"]
                self.model.save(str(self.out_dir / "best_model"))
                normalizer = self.model.get_vec_normalize_env()
                if normalizer is not None:
                    normalizer.save(
                        str(self.out_dir / "best_vecnorm.pkl"))
                result["source"] = "deterministic evaluation"
                (self.out_dir / "best_metrics.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                self.logger.record(
                    "level3_eval/best_score", self.best_score)

        if not self.misses:
            return True

        misses = np.asarray(self.misses)
        falls = np.asarray(self.falls)
        distances = np.asarray(self.distances)
        during_distances = np.asarray(self.during_distances)
        line_errors = np.asarray(self.line_errors)
        score = float(
            misses.mean()
            + 5.0 * falls.mean()
            + np.maximum(0.0, 2.0 - distances).mean()
            + 3.0 * np.maximum(
                0.0, 0.4 - during_distances).mean()
            + 2.0 * np.maximum(
                0.0, line_errors - 0.2).mean()
        )
        self.logger.record("level3/mean_miss", float(misses.mean()))
        self.logger.record("level3/fall_rate", float(falls.mean()))
        self.logger.record(
            "level3/mean_distance_walked", float(distances.mean()))
        self.logger.record(
            "level3/mean_distance_during_throw",
            float(during_distances.mean()))
        self.logger.record(
            "level3/mean_max_abs_lateral_error",
            float(line_errors.mean()))
        self.logger.record(
            "level3/hit_rate_0.50m", float((misses < 0.5).mean()))
        self.logger.record("level3/selection_score", score)

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs=2, type=float, default=[4.0, 0.0],
                        metavar=("X", "Y"))
    parser.add_argument("--random-target", action="store_true")
    parser.add_argument("--target-r", nargs=2, type=float,
                        default=[3.0, 5.0], metavar=("MIN", "MAX"))
    parser.add_argument("--target-theta", nargs=2, type=float,
                        default=[-0.25, 0.25], metavar=("MIN", "MAX"))
    parser.add_argument("--command", nargs=3, type=float,
                        default=[1.3, 0.0, 0.0],
                        metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda",
                        help="TD3 training device; default is NVIDIA CUDA")
    parser.add_argument("--walk-policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--walk-policy-device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="CPU is faster for this tiny 50 Hz LSTM")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--stage", choices=["throw", "full"],
                        default="throw",
                        help="train throw first, then resume with full")
    parser.add_argument("--learning-starts", type=int, default=0,
                        help="keep 0: random warmup destroys pretrained gait")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="conservative fine-tuning around strong baseline")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4,
                        help="parallel CPU MuJoCo environments")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"],
                        default="subproc")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--throw-start", type=float, default=1.4)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a GPU. "
            "Run this script in the mujo_torch conda environment.")
    torch.set_float32_matmul_precision("high")

    target = [float(value) for value in args.target]
    command = [float(value) for value in args.command]
    name = args.name or default_name(target, command)
    out = Path(__file__).resolve().parent / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    actual_device = (
        torch.cuda.get_device_name(0)
        if args.device.startswith("cuda") else args.device
    )
    project_root = Path(__file__).resolve().parent.parent
    resolved_walk_policy = Path(args.walk_policy).resolve()
    try:
        stored_walk_policy = str(
            resolved_walk_policy.relative_to(project_root))
    except ValueError:
        stored_walk_policy = str(resolved_walk_policy)

    config = {
        "algorithm": "TD3",
        "task": "walk forward while throwing to a marked target",
        "target": target,
        "random_target": args.random_target,
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "command": command,
        "throw_start": args.throw_start,
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
        "action_groups": ACTION_GROUPS,
        "action_count": N_ACTIONS,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "seed": args.seed,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    env_kwargs = {
        "fixed_target": None if args.random_target else target,
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "command": command,
        "throw_start": args.throw_start,
        "stage": args.stage,
        "max_seconds": args.max_seconds,
        "walk_policy_path": args.walk_policy,
        "walk_policy_device": args.walk_policy_device,
    }
    env_fns = [
        make_env(rank, seed=args.seed, **env_kwargs)
        for rank in range(args.num_envs)
    ]
    if args.vec_env == "subproc" and args.num_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    resume_name = (
        Path(args.resume).stem if args.resume is not None else "")
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
            env, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0)

    sigma = np.full(N_ACTIONS, 0.07, dtype=np.float32)
    sigma[0:12] = 0.05
    sigma[23] = 0.12
    if args.stage == "throw":
        sigma[:19] = 0.0
    noise = NormalActionNoise(
        np.zeros(N_ACTIONS, dtype=np.float32), sigma)

    if args.resume:
        model = TD3.load(
            args.resume, env=env, device=args.device,
            tensorboard_log=str(out / "tb"))
        # Loading restores the checkpoint's optimizer learning rate. Explicitly
        # replace it so stage-2 fine-tuning obeys this command line.
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        for optimizer in (model.actor.optimizer, model.critic.optimizer):
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        model.action_noise = noise
        reset_num_timesteps = False
        metrics_path = out / "best_metrics.json"
        callback_initial_best = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            ["selection_score"]
            if metrics_path.exists() else np.inf
        )
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
            target_policy_noise=0.15,
            target_noise_clip=0.35,
            policy_kwargs={"net_arch": [512, 512]},
            tensorboard_log=str(out / "tb"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        zero_initialise_actor(model)
        reset_num_timesteps = True

        initial = evaluate_once(model, env)
        model.save(str(out / "initial_model"))
        env.save(str(out / "initial_vecnorm.pkl"))
        model.save(str(out / "best_model"))
        env.save(str(out / "best_vecnorm.pkl"))
        callback_initial_best = selection_score(initial)
        (out / "best_metrics.json").write_text(
            json.dumps({
                "selection_score": callback_initial_best,
                "mean_miss": float(initial["objective_miss"]),
                "fall_rate": float(initial["fallen"]),
                "mean_distance_walked": float(
                    initial["distance_walked"]),
                "distance_during_throw": float(
                    initial["distance_during_throw"]),
                "positive_velocity_fraction_during_throw": float(
                    initial[
                        "positive_velocity_fraction_during_throw"]),
                "max_abs_lateral_error": float(
                    initial["max_abs_lateral_error"]),
                "source": "zero-residual initial controller",
            }, indent=2),
            encoding="utf-8",
        )
        print(
            "initial controller: "
            f"miss={initial['objective_miss']:.3f} m, "
            f"walked={initial['distance_walked']:.3f} m, "
            f"during_throw={initial['distance_during_throw']:.3f} m, "
            f"line_error={initial['max_abs_lateral_error']:.3f} m, "
            f"fallen={initial['fallen']}"
        )

    print(f"TD3 training device: {actual_device}")
    print(
        f"Walking LSTM device: {args.walk_policy_device} "
        "(50 Hz inference)"
    )

    callbacks = [
        Level3Callback(
            out,
            eval_kwargs=env_kwargs,
            initial_best=callback_initial_best,
        ),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
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
        env.close()
        print(f"saved model and normalizer to {out}")


if __name__ == "__main__":
    main()

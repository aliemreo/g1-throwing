"""Train TD3 to throw the ball to one specified XY target.

Examples (run from this directory):

    python train_td3.py --target 4 0 --steps 1_000_000
    tensorboard --logdir runs/td3_target_4_0/tb

The reward has one and only one objective: reduce the ball's landing error.
See env.py for the exact telescoping reward definition.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import mujoco
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_env import make_env


# These are the joints used by the verified hand-written throw in check_env.py.
# All other actuators stay at the standing keyframe pose.
THROW_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
)

# Verified trajectory used as the nominal controller for residual TD3. A zero
# policy already performs a real throw; TD3 learns continuous corrections that
# reduce the same target-error reward used everywhere else.
WINDUP = {
    "right_shoulder_pitch_joint": 1.735366,
    "right_shoulder_roll_joint": -0.711473,
    "right_elbow_joint": 1.913143,
}
RELEASE = {
    "right_shoulder_pitch_joint": -1.795524,
    "right_shoulder_roll_joint": -0.625955,
    "right_elbow_joint": 0.073054,
}
T_WINDUP = 0.600418
T_SWING = 0.889800
T_LET_GO = 0.784
RELEASE_TIME_RANGE = (0.72, 0.84)
RESIDUAL_SCALE = (0.08, 0.08, 0.08, 0.02)


def _smooth(start, end, phase):
    phase = np.clip(phase, 0.0, 1.0)
    return start + (end - start) * 0.5 * (1.0 - np.cos(np.pi * phase))


def warmstart_action(env, time_s):
    """Reduced four-dimensional action for the verified throwing motion."""
    action = np.zeros(len(THROW_JOINTS) + 1, dtype=np.float32)
    for out_idx, name in enumerate(THROW_JOINTS):
        actuator_id = mujoco.mj_name2id(
            env.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if time_s < T_WINDUP:
            angle = _smooth(env.q_stand[actuator_id], WINDUP[name],
                            time_s / T_WINDUP)
        else:
            angle = _smooth(WINDUP[name], RELEASE[name],
                            (time_s - T_WINDUP) / (T_SWING - T_WINDUP))
        action[out_idx] = np.clip(
            (angle - env.q_stand[actuator_id]) / env.scale[actuator_id],
            -1.0, 1.0)

    lo, hi = RELEASE_TIME_RANGE
    action[-1] = 2.0 * (T_LET_GO - lo) / (hi - lo) - 1.0
    return action


def zero_initialise_actor(model):
    """Make the initial residual policy exactly zero (the nominal throw)."""
    output_layer = next(
        layer for layer in reversed(list(model.actor.mu.modules()))
        if isinstance(layer, torch.nn.Linear))
    torch.nn.init.zeros_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    model.actor_target.load_state_dict(model.actor.state_dict())


def evaluate_once(model, venv):
    """Return deterministic objective miss from one fixed-target episode."""
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        if done[0]:
            return float(infos[0]["objective_miss"])


class AccuracyCallback(BaseCallback):
    """Log the real final target error and retain the best policy."""

    def __init__(self, out_dir, window=100, initial_best=np.inf):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.window = window
        self.misses = deque(maxlen=window)
        self.best_mean = float(initial_best)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            miss = info.get("episode_objective_miss")
            if miss is not None:
                self.misses.append(float(miss))

        if self.misses:
            values = np.asarray(self.misses)
            mean = float(values.mean())
            self.logger.record("throw/mean_miss", mean)
            self.logger.record("throw/median_miss", float(np.median(values)))
            self.logger.record("throw/hit_rate_0.50m",
                               float((values < 0.50).mean()))
            self.logger.record("throw/hit_rate_0.25m",
                               float((values < 0.25).mean()))

            # Require a full window so an unusually lucky first throw is not
            # selected as the best policy.
            if len(self.misses) == self.window and mean < self.best_mean:
                self.best_mean = mean
                self.model.save(str(self.out_dir / "best_model"))
                vecnorm = self.model.get_vec_normalize_env()
                if vecnorm is not None:
                    vecnorm.save(str(self.out_dir / "best_vecnorm.pkl"))
                self.logger.record("throw/best_mean_miss", self.best_mean)
        return True


def default_name(target):
    def clean(value):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return f"td3_target_{clean(target[0])}_{clean(target[1])}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=2, default=[4.0, 0.0],
                        metavar=("X", "Y"), help="fixed target in metres")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="TD3 .zip checkpoint to continue")
    parser.add_argument("--learning-starts", type=int, default=None,
                        help="random-action steps; default 1000 with warm-start, "
                             "10000 with --no-warmstart")
    parser.add_argument("--no-warmstart", action="store_true",
                        help="disable the verified nominal throw")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    learning_starts = (args.learning_starts if args.learning_starts is not None
                       else (10_000 if args.no_warmstart else 1_000))

    name = args.name or default_name(args.target)
    out = Path(__file__).parent / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "algorithm": "TD3",
        "target": [float(x) for x in args.target],
        "action_joints": list(THROW_JOINTS),
        "release_time_range": list(RELEASE_TIME_RANGE),
        "residual_baseline": not args.no_warmstart,
        "residual_scale": list(RESIDUAL_SCALE),
        "seed": args.seed,
        "reward": "landing-error reduction only",
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    env_kwargs = dict(
        fixed_target=args.target,
        freeze_legs=True,
        action_joints=THROW_JOINTS,
        release_time_range=RELEASE_TIME_RANGE,
        action_baseline=None if args.no_warmstart else warmstart_action,
        action_residual_scale=RESIDUAL_SCALE,
    )
    venv = DummyVecEnv([make_env(0, seed=args.seed, **env_kwargs)])

    resume_vecnorm = out / "vecnorm.pkl"
    if args.resume and resume_vecnorm.exists():
        venv = VecNormalize.load(str(resume_vecnorm), venv)
        venv.training = True
        venv.norm_reward = False
    else:
        venv = VecNormalize(
            venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    n_actions = venv.action_space.shape[-1]
    # The release channel is deliberately noisier because release timing is
    # the most sensitive part of the throw.
    sigma = np.full(n_actions, 0.08, dtype=np.float32)
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)

    initial_best = np.inf
    if args.resume:
        model = TD3.load(
            args.resume, env=venv, device=args.device,
            tensorboard_log=str(out / "tb"))
        model.action_noise = action_noise
        reset_num_timesteps = False
        print(f"resumed from {args.resume}")
    else:
        model = TD3(
            "MlpPolicy",
            venv,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=learning_starts,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=4,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out / "tb"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        reset_num_timesteps = True

        if not args.no_warmstart:
            zero_initialise_actor(model)
            initial_best = evaluate_once(model, venv)
            model.save(str(out / "best_model"))
            venv.save(str(out / "best_vecnorm.pkl"))
            print(f"nominal deterministic miss={initial_best:.3f} m")

    callbacks = [
        AccuracyCallback(out, initial_best=initial_best),
        CheckpointCallback(
            save_freq=100_000,
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
        venv.save(str(out / "vecnorm.pkl"))
        venv.close()
        print(f"\nsaved TD3 policy and normalisation statistics to {out}")


if __name__ == "__main__":
    main()

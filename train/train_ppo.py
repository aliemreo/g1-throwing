"""
PPO training for the G1 throwing task.

    python train_ppo.py --steps 10_000_000 --envs 8

Windows note: SubprocVecEnv spawns fresh interpreters, so everything must sit
behind the __main__ guard at the bottom. Without it you get an infinite
process fork bomb.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gym_env import make_env


class MissLogger(BaseCallback):
    """Logs mean landing error - the metric that actually matters.

    Reward is not comparable across episodes: a far target gives the shaping
    term more room to accumulate, so mean reward conflates difficulty with
    skill. Miss distance does not.
    """

    def __init__(self, window=100):
        super().__init__()
        self.misses = []
        self.no_release = 0
        self.episodes = 0
        self.window = window

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode_miss" in info:
                self.misses.append(info["episode_miss"])
                self.episodes += 1
            elif info.get("released") is False and info.get("landed") is None:
                pass  # only counted at episode end below

        if len(self.misses) >= self.window:
            m = np.array(self.misses[-self.window:])
            self.logger.record("throw/mean_miss", float(m.mean()))
            self.logger.record("throw/median_miss", float(np.median(m)))
            self.logger.record("throw/hit_rate_1m", float((m < 1.0).mean()))
            self.logger.record("throw/hit_rate_0.5m", float((m < 0.5).mean()))
            self.misses = self.misses[-self.window:]
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--name", default="throw_v1")
    ap.add_argument("--resume", default=None, help="path to a saved .zip")
    ap.add_argument("--easy", action="store_true",
                    help="narrow target range - use this first")
    args = ap.parse_args()

    out = Path(__file__).parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    # Curriculum: start narrow. Widen once mean_miss drops below ~0.8 m.
    env_kwargs = dict(target_r=(3.5, 4.5), target_theta=(-0.15, 0.15)) \
        if args.easy else dict(target_r=(2.5, 6.0), target_theta=(-0.5, 0.5))

    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    venv = vec_cls([make_env(i, seed=0, **env_kwargs) for i in range(args.envs)])

    # Normalise observations: joint angles, velocities, and target positions
    # live on very different scales. Do NOT normalise reward - the terminal
    # bonus carries the signal and rescaling it distorts the shaping balance.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    n_steps = 1024                       # per env -> 8192 samples per update
    model_kwargs = dict(
        policy="MlpPolicy",
        env=venv,
        n_steps=n_steps,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        # gamma=1.0 matches the shaping term in env.py, which makes the
        # potential-based shaping exactly policy-invariant. Episodes always
        # terminate within 300 steps, so an undiscounted return is finite.
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,                  # release timing needs exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(out / "tb"),
        verbose=1,
    )

    if args.resume:
        model = PPO.load(args.resume, env=venv, **{
            k: v for k, v in model_kwargs.items()
            if k not in ("policy", "env")})
        print(f"resumed from {args.resume}")
    else:
        model = PPO(**model_kwargs)

    callbacks = [
        MissLogger(),
        CheckpointCallback(save_freq=max(200_000 // args.envs, 1),
                           save_path=str(out), name_prefix="ckpt"),
    ]

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    progress_bar=True)
    finally:
        model.save(str(out / "final"))
        venv.save(str(out / "vecnorm.pkl"))
        venv.close()
        print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
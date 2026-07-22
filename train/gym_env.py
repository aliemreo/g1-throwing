"""
Gymnasium wrapper around ThrowEnv.

ThrowEnv already uses the gymnasium call signature; this adds the spaces,
float32 casting, and seeding that stable-baselines3 requires.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import ThrowEnv


class ThrowGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.env = ThrowEnv(**kwargs)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.n_act,), dtype=np.float32)

        # observations are unbounded in principle; VecNormalize handles scaling
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # surface the metric we actually care about for logging
        if terminated and info.get("miss") is not None:
            info["episode_miss"] = float(info["miss"])

        return (obs.astype(np.float32), float(reward),
                bool(terminated), bool(truncated), info)


def make_env(rank=0, seed=0, **kwargs):
    """Factory for SubprocVecEnv. Must be picklable, hence module level."""
    def _init():
        return ThrowGym(seed=seed + rank, **kwargs)
    return _init
"""Gymnasium wrapper and picklable factory for WalkingThrowEnv."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from walking_throw_env import WalkingThrowEnv


class WalkingThrowGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.env = WalkingThrowEnv(**kwargs)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.env.n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.env.n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed)
        return observation.astype(np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        if terminated or truncated:
            info["episode_objective_miss"] = float(
                info["objective_miss"])
            info["episode_fallen"] = float(info["fallen"])
            info["episode_distance_walked"] = float(
                info["distance_walked"])
            info["episode_lateral_drift"] = float(
                info["lateral_drift"])
            info["episode_max_abs_lateral_error"] = float(
                info["max_abs_lateral_error"])
            info["episode_distance_during_throw"] = float(
                info["distance_during_throw"])
            info["episode_positive_velocity_fraction_during_throw"] = float(
                info["positive_velocity_fraction_during_throw"])
            info["episode_landed"] = float(
                info["landing_miss"] is not None)
        return (
            observation.astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def make_env(rank=0, seed=0, **kwargs):
    def _init():
        return WalkingThrowGym(seed=seed + rank, **kwargs)
    return _init

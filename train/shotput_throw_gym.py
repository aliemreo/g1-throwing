"""Gymnasium wrapper for the learned walking shot-put environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shotput_throw_env import ShotPutWalkingEnv


class ShotPutWalkingGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.env = ShotPutWalkingEnv(**kwargs)
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.env.n_act,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.env.n_obs,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed)
        return observation.astype(np.float32), info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        if terminated or truncated:
            info.update({
                "episode_objective_miss": float(
                    info["objective_miss"]),
                "episode_fallen": float(info["fallen"]),
                "episode_released": float(info["released"]),
                "episode_landed": float(
                    info["landing_miss"] is not None),
                "episode_style_valid": float(
                    info["shotput_style_valid"]),
                "episode_style_score": float(
                    info["shotput_style_score"]),
                "episode_ready_error": float(
                    info["shotput_ready_error"]),
                "episode_release_forward_speed": float(
                    info["release_forward_speed"]),
                "episode_release_hand_height": float(
                    info["release_hand_height"]),
                "episode_release_elbow_angle": float(
                    info["release_elbow_angle"]),
                "episode_distance_walked": float(
                    info["distance_walked"]),
                "episode_lateral_drift": float(
                    info["lateral_drift"]),
                "episode_max_abs_lateral_error": float(
                    info["max_abs_lateral_error"]),
                "episode_distance_during_throw": float(
                    info["distance_during_throw"]),
                "episode_positive_velocity_fraction_during_throw": float(
                    info[
                        "positive_velocity_fraction_during_throw"]),
            })
        return (
            observation.astype(np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def make_shotput_env(rank=0, seed=0, **kwargs):
    def _init():
        return ShotPutWalkingGym(seed=seed + rank, **kwargs)
    return _init

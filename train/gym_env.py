"""
Gymnasium wrapper around ThrowEnv.

ThrowEnv already uses the gymnasium call signature; this adds the spaces,
float32 casting, and seeding that stable-baselines3 requires.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from env import ThrowEnv


class ThrowGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_joints=None, action_baseline=None,
                 action_residual_scale=1.0, **kwargs):
        super().__init__()
        self.env = ThrowEnv(**kwargs)
        self.action_baseline = action_baseline

        # TD3 learns much faster when it does not have to explore joints that
        # are irrelevant to a pinned-base, one-arm throw. The raw environment
        # remains full-dimensional for the existing PPO code.
        self.action_ids = None
        if action_joints is not None:
            ids = []
            for name in action_joints:
                idx = mujoco.mj_name2id(
                    self.env.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if idx < 0:
                    raise ValueError(f"unknown actuator: {name}")
                ids.append(idx)
            if len(set(ids)) != len(ids):
                raise ValueError("action_joints contains duplicates")
            self.action_ids = np.asarray(ids, dtype=np.int32)

        n_actions = (self.env.n_act if self.action_ids is None
                     else len(self.action_ids) + 1)
        self.action_residual_scale = np.broadcast_to(
            np.asarray(action_residual_scale, dtype=np.float32),
            (n_actions,)).copy()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # observations are unbounded in principle; VecNormalize handles scaling
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.n_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.action_baseline is not None:
            time_s = self.env.step_i * self.env.dt
            baseline = np.asarray(
                self.action_baseline(self.env, time_s), dtype=np.float32)
            if baseline.shape != action.shape:
                raise ValueError(
                    f"baseline shape {baseline.shape} "
                    f"!= action shape {action.shape}")
            action = np.clip(
                baseline + action * self.action_residual_scale, -1.0, 1.0)

        if self.action_ids is None:
            full_action = action
        else:
            full_action = np.zeros(self.env.n_act, dtype=np.float32)
            full_action[self.action_ids] = action[:-1]
            full_action[-1] = action[-1]

        obs, reward, terminated, truncated, info = self.env.step(full_action)

        # surface the metric we actually care about for logging
        if terminated and info.get("miss") is not None:
            info["episode_miss"] = float(info["miss"])
        if (terminated or truncated) and info.get("objective_miss") is not None:
            info["episode_objective_miss"] = float(info["objective_miss"])

        return (obs.astype(np.float32), float(reward),
                bool(terminated), bool(truncated), info)


def make_env(rank=0, seed=0, **kwargs):
    """Factory for SubprocVecEnv. Must be picklable, hence module level."""
    def _init():
        return ThrowGym(seed=seed + rank, **kwargs)
    return _init

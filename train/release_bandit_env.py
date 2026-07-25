"""One-step contextual-bandit view of the release-timing aiming task.

In the multi-step task the release action affects the outcome at only ~1 of
~250 steps, so its learning gradient was diluted ~250x and TD3 collapsed to a
constant, target-independent throw. Here the whole walk+throw is rolled out
*inside* a single env step, so every RL transition is exactly
    (target context) -> (release timing action) -> (landing reward),
with no dilution. The agent then learns release = f(target) directly.

Observation : the target distance, centred near the 4 m nominal (1-D).
Action      : the release timing in [-1, 1] (maps, via the re-centred window, to
              the full 3.3-5 m landing band; the rest of the body throws the
              fixed nominal motion, so no residual can destabilise the robot).
Reward      : -landing_miss (minus a fall penalty as a safety net).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from walking_throw_env import N_ACTIONS, WalkingThrowEnv

RELEASE_INDEX = 23


class ReleaseBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        target_r=(3.3, 5.0),
        target_theta=(0.0, 0.0),
        fixed_target=None,
        release_window=0.0055,
        release_center_offset=0.781,
        max_seconds=3.5,
        fall_penalty=5.0,
        seed=None,
        **inner_kwargs,
    ):
        super().__init__()
        # stage="release" freezes every residual except the release timing, so
        # the throw is the stable nominal motion and only the release varies.
        self.inner = WalkingThrowEnv(
            fixed_target=fixed_target,
            target_r=target_r,
            target_theta=target_theta,
            release_window=release_window,
            release_center_offset=release_center_offset,
            stage="release",
            max_seconds=max_seconds,
            seed=seed,
            **inner_kwargs,
        )
        self.fall_penalty = float(fall_penalty)
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32)

    def _context(self):
        distance = float(np.linalg.norm(self.inner.target[:2]))
        return np.array([distance - 4.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Not passing a seed keeps the inner rng advancing, so each episode draws
        # a fresh random target (unless the inner env has a fixed_target).
        self.inner.reset(seed=seed)
        return self._context(), {}

    def step(self, action):
        release = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        inner_action = np.zeros(N_ACTIONS, dtype=np.float64)
        inner_action[RELEASE_INDEX] = release

        info = {}
        while True:
            _, _, terminated, truncated, info = self.inner.step(inner_action)
            landed = self.inner.landed is not None
            # Stop as soon as the ball lands (or the robot falls / time runs
            # out) - the walking follow-through does not affect the reward.
            if terminated or truncated or landed:
                break

        if self.inner.landed is not None:
            miss = float(np.linalg.norm(
                self.inner.landed[:2] - self.inner.target[:2]))
            throw_distance = float(self.inner.landed[0])
        else:
            miss = float(np.linalg.norm(
                self.inner._predicted_landing() - self.inner.target[:2]))
            throw_distance = float("nan")
        fallen = bool(info["fallen"])
        reward = -miss - (self.fall_penalty if fallen else 0.0)

        out_info = {
            "objective_miss": miss,
            "fallen": fallen,
            "target": self.inner.target[:2].copy(),
            "release_action": release,
            "throw_distance": throw_distance,
            "landed": (
                None if self.inner.landed is None
                else self.inner.landed.copy()),
            # episode_* keys so a simple logger can pick them up on done.
            "episode_objective_miss": miss,
            "episode_fallen": float(fallen),
            "episode_throw_distance": throw_distance,
        }
        # A one-step episode: terminated and truncated are both true.
        return self._context(), float(reward), True, True, out_info


def make_bandit_env(rank=0, seed=0, **kwargs):
    def _init():
        return ReleaseBanditEnv(seed=seed + rank, **kwargs)
    return _init

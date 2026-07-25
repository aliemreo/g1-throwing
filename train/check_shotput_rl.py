"""Fast structural checks for the learned shot-put environment."""

import numpy as np

from shotput_throw_env import (
    SHOTPUT_N_ACTIONS,
    ShotPutWalkingEnv,
)


def rollout(action_function, seed=123):
    env = ShotPutWalkingEnv(
        fixed_target=(4.0, 0.0),
        command=(1.3, 0.0, 0.0),
        max_seconds=3.8,
        stage="arm",
        seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    initial_right_target = env.right_arm_target.copy()
    for step in range(env.max_steps):
        action = np.asarray(
            action_function(step, observation),
            dtype=np.float32,
        )
        observation, _, terminated, truncated, info = env.step(
            action)
        if terminated or truncated:
            return env, initial_right_target, info
    raise AssertionError("rollout did not terminate")


def main():
    zero = np.zeros(SHOTPUT_N_ACTIONS, dtype=np.float32)
    zero_env, neutral, zero_info = rollout(
        lambda _step, _observation: zero)

    if not np.allclose(
        zero_env.right_arm_target,
        neutral,
        atol=1e-5,
    ):
        raise AssertionError(
            "zero actor unexpectedly contains an arm trajectory")
    if zero_info["shotput_style_valid"]:
        raise AssertionError(
            "untrained zero actor must not pass the shot-put style gate")
    if "no nominal throw trajectory" not in (
        zero_info["controller_source"]
    ):
        raise AssertionError("controller source audit is missing")

    varied = np.zeros(SHOTPUT_N_ACTIONS, dtype=np.float32)
    varied[19:23] = [0.65, -0.50, 0.0, 0.50]
    varied[23] = -0.25
    varied_env, varied_neutral, _ = rollout(
        lambda _step, _observation: varied,
        seed=456,
    )
    if np.allclose(
        varied_env.right_arm_target,
        varied_neutral,
        atol=1e-3,
    ):
        raise AssertionError(
            "TD3 right-arm actions do not affect joint targets")

    print("shot-put RL environment checks passed")
    print(f"observation size : {zero_env.n_obs}")
    print(f"action size      : {zero_env.n_act}")
    print(
        "zero actor style : "
        f"{zero_info['shotput_style_score']:.3f}")
    print(
        "zero actor valid : "
        f"{zero_info['shotput_style_valid']}")
    print(
        "zero actor miss  : "
        f"{zero_info['objective_miss']:.3f} m")
    print("nominal trajectory: NONE")


if __name__ == "__main__":
    main()

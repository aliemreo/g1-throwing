"""Fast deterministic integration check for the zero-residual controller."""

import argparse

import numpy as np

from walking_throw_env import WalkingThrowEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", nargs=3, type=float,
                        default=[1.3, 0.0, 0.0])
    parser.add_argument("--target", nargs=2, type=float,
                        default=[4.0, 0.0])
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    env = WalkingThrowEnv(
        command=args.command,
        fixed_target=args.target,
        max_seconds=args.seconds,
        seed=0,
    )
    observation, _ = env.reset()
    if observation.shape != (env.n_obs,):
        raise RuntimeError("observation size changed during reset")

    info = None
    total_reward = 0.0
    for _ in range(env.max_steps):
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(env.n_act, dtype=np.float32))
        if not np.isfinite(observation).all():
            raise RuntimeError("non-finite observation")
        total_reward += reward
        if terminated or truncated:
            break

    print(f"observation size : {env.n_obs}")
    print(f"action size      : {env.n_act}")
    print(f"simulated time   : {env.d.time:.3f} s")
    print(f"walked distance  : {info['distance_walked']:.3f} m")
    print(f"lateral drift    : {info['lateral_drift']:.3f} m")
    print(f"max line error   : {info['max_abs_lateral_error']:.3f} m")
    print(f"before throw     : {info['distance_before_throw']:.3f} m")
    print(f"during throw     : {info['distance_during_throw']:.3f} m")
    print(f"after throw      : {info['distance_after_throw']:.3f} m")
    print(
        "moving in throw  : "
        f"{info['positive_velocity_fraction_during_throw'] * 100:.1f}%"
    )
    print(
        "min throw vx     : "
        f"{info['min_forward_velocity_during_throw']:.3f} m/s"
    )
    print(f"released         : {info['released']}")
    print(f"landed           : {info['landing_miss'] is not None}")
    print(f"objective miss   : {info['objective_miss']:.3f} m")
    print(f"fallen           : {info['fallen']}")
    print(f"total reward     : {total_reward:.3f}")


if __name__ == "__main__":
    main()

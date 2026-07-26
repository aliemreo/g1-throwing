"""Watch the walk-and-throw controller aim at different targets in MuJoCo.

Three modes:
    --mode rl        (default) the trained RL policy chooses the release timing.
    --mode baseline  no RL: the fixed nominal release (throws ~4 m for every
                     target - it cannot aim).
    --mode compare   for each target, throw the no-RL baseline first, then the
                     RL policy - so you SEE the baseline miss and the RL hit.

    python view_bandit.py --run bandit_bigball --mode compare
    python view_bandit.py --run bandit_bigball --mode compare --targets 3.3 5.0
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import TD3

from release_bandit_env import RELEASE_INDEX
from walking_throw_env import N_ACTIONS, WalkingThrowEnv


def build_shots(targets, mode):
    """Ordered list of (target, kind) shots to render, cycled forever."""
    if mode == "compare":
        shots = []
        for target in targets:
            shots.append((target, "baseline"))
            shots.append((target, "rl"))
        return shots
    return [(target, mode) for target in targets]


def release_action(kind, distance, model):
    """24-D action: only the release channel is set. baseline = 0 (nominal)."""
    action = np.zeros(N_ACTIONS, dtype=np.float64)
    if kind == "rl":
        obs = np.array([[distance - 4.0]], dtype=np.float32)
        release, _ = model.predict(obs, deterministic=True)
        action[RELEASE_INDEX] = float(release[0][0])
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="best_model")
    parser.add_argument("--mode", choices=["rl", "baseline", "compare"],
                        default="rl")
    parser.add_argument("--targets", nargs="*", type=float,
                        default=[3.3, 3.7, 4.1, 4.5, 5.0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cam", default="follow")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8"))
    model = TD3.load(str(run_dir / args.ckpt), device=args.device)

    env = WalkingThrowEnv(
        stage="release",
        release_window=config.get("release_window", 0.0055),
        release_center_offset=config.get("release_center_offset", 0.781),
        max_seconds=5.0,
    )

    shots = build_shots(list(args.targets), args.mode)
    label = {"baseline": "NO-RL baseline", "rl": "RL policy"}
    index = 0

    def setup(shot):
        target, kind = shot
        env.fixed_target = np.asarray([target, 0.0], dtype=np.float64)
        env.reset()
        distance = float(np.linalg.norm(env.target[:2]))
        return release_action(kind, distance, model)

    action = setup(shots[0])

    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        camera_id = mujoco.mj_name2id(
            env.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            _, _, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.dt)
            if terminated or truncated:
                target, kind = shots[index]
                threw = (f"{info['landed'][0]:.2f} m"
                         if info["landed"] is not None else "n/a")
                hit = "HIT " if info["objective_miss"] < 0.5 else "miss"
                print(
                    f"[{label[kind]:>14}]  target={target:.2f} m  ->  "
                    f"threw to {threw}  ({hit} by "
                    f"{info['objective_miss']:.2f} m)")
                time.sleep(1.0)
                index = (index + 1) % len(shots)
                action = setup(shots[index])


if __name__ == "__main__":
    main()

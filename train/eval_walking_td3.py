"""Evaluate or view a trained walking-and-throwing TD3 controller."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from walking_throw_env import WalkingThrowEnv
from walking_throw_gym import make_env


def load_run(run_dir, target_override):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = target_override or config["target"]
    walk_policy = Path(config["walk_policy"])
    if not walk_policy.is_absolute():
        walk_policy = (
            Path(__file__).resolve().parent.parent / walk_policy)
    kwargs = {
        "fixed_target": (
            None if config.get("random_target", False) else target),
        "target_r": config.get("target_r", [3.0, 5.0]),
        "target_theta": config.get("target_theta", [-0.25, 0.25]),
        "command": config.get("command", [1.3, 0.0, 0.0]),
        "throw_start": config.get("throw_start", 1.4),
        "stage": config.get("stage", "throw"),
        "max_seconds": config.get("max_seconds", 5.0),
        "walk_policy_path": str(walk_policy),
        "walk_policy_device": "cpu",
    }
    if target_override is not None:
        kwargs["fixed_target"] = target
    return config, target, kwargs


def find_checkpoint(run_dir, name):
    path = run_dir / name
    if path.with_suffix(".zip").exists():
        return path
    if name == "best_model":
        fallback = run_dir / "final"
        if fallback.with_suffix(".zip").exists():
            return fallback
    raise FileNotFoundError(f"checkpoint not found: {path}.zip")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="run folder name under train/runs")
    parser.add_argument("--ckpt", default="best_model",
                        help="checkpoint name without .zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target", nargs=2, type=float, default=None,
                        metavar=("X", "Y"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--cam", default="follow")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    _, target, env_kwargs = load_run(run_dir, args.target)
    checkpoint = find_checkpoint(run_dir, args.ckpt)
    model = TD3.load(str(checkpoint), device=args.device)

    normalizer = (
        run_dir / "best_vecnorm.pkl"
        if checkpoint.name == "best_model"
        else run_dir / "vecnorm.pkl"
    )
    if not normalizer.exists():
        normalizer = run_dir / "initial_vecnorm.pkl"
    vec_env = DummyVecEnv([make_env(
        0, seed=123, **env_kwargs)])
    vec_env = VecNormalize.load(str(normalizer), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    if not args.view:
        misses, distances, drifts, max_line_errors = [], [], [], []
        before, during, after, positive_fractions = [], [], [], []
        min_throw_velocities, falls, landed = [], [], []
        for _ in range(args.episodes):
            observation = vec_env.reset()
            while True:
                action, _ = model.predict(
                    observation, deterministic=True)
                observation, _, done, infos = vec_env.step(action)
                if done[0]:
                    info = infos[0]
                    misses.append(info["objective_miss"])
                    distances.append(info["distance_walked"])
                    drifts.append(abs(info["lateral_drift"]))
                    max_line_errors.append(
                        info["max_abs_lateral_error"])
                    before.append(info["distance_before_throw"])
                    during.append(info["distance_during_throw"])
                    after.append(info["distance_after_throw"])
                    positive_fractions.append(
                        info[
                            "positive_velocity_fraction_during_throw"])
                    min_throw_velocities.append(
                        info["min_forward_velocity_during_throw"])
                    falls.append(info["fallen"])
                    landed.append(info["landing_miss"] is not None)
                    break

        misses = np.asarray(misses, dtype=float)
        distances = np.asarray(distances, dtype=float)
        falls = np.asarray(falls, dtype=float)
        print(f"checkpoint      : {checkpoint.with_suffix('.zip')}")
        print(f"target          : ({target[0]:.2f}, {target[1]:.2f}) m")
        print(f"episodes        : {args.episodes}")
        print(f"mean miss       : {misses.mean():.3f} m")
        print(f"within 0.50 m   : {(misses < 0.50).mean() * 100:.1f}%")
        print(f"mean distance   : {distances.mean():.3f} m")
        print(f"before throw    : {np.mean(before):.3f} m")
        print(f"during throw    : {np.mean(during):.3f} m")
        print(f"after throw     : {np.mean(after):.3f} m")
        print(
            "moving in throw : "
            f"{np.mean(positive_fractions) * 100:.1f}%")
        print(
            "min throw vx    : "
            f"{np.mean(min_throw_velocities):.3f} m/s")
        print(f"final abs drift : {np.mean(drifts):.3f} m")
        print(
            "max line error  : "
            f"{np.mean(max_line_errors):.3f} m")
        print(f"fall rate       : {falls.mean() * 100:.1f}%")
        print(f"ball landed     : {sum(landed)}/{args.episodes}")
        vec_env.close()
        return

    raw = WalkingThrowEnv(seed=123, **env_kwargs)
    observation, _ = raw.reset()
    with mujoco.viewer.launch_passive(raw.m, raw.d) as viewer:
        camera_id = mujoco.mj_name2id(
            raw.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            normalized = vec_env.normalize_obs(
                observation[None].astype(np.float32))
            action, _ = model.predict(normalized, deterministic=True)
            observation, _, terminated, truncated, info = raw.step(
                action[0])
            viewer.sync()
            time.sleep(raw.dt)
            if terminated or truncated:
                print(
                    f"miss={info['objective_miss']:.3f} m, "
                    f"walked={info['distance_walked']:.3f} m, "
                    f"during={info['distance_during_throw']:.3f} m, "
                    f"drift={info['lateral_drift']:.3f} m, "
                    f"fallen={info['fallen']}"
                )
                time.sleep(0.7)
                observation, _ = raw.reset()
    vec_env.close()


if __name__ == "__main__":
    main()

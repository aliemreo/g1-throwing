"""Evaluate or view a TD3 throwing policy."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import ThrowEnv
from gym_env import make_env
from train_td3 import (RELEASE_TIME_RANGE, RESIDUAL_SCALE, THROW_JOINTS,
                       default_name, warmstart_action)


def load_config(out, target_override):
    path = out / "config.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    target = target_override or config.get("target", [4.0, 0.0])
    joints = tuple(config.get("action_joints", THROW_JOINTS))
    release_range = tuple(config.get("release_time_range", RELEASE_TIME_RANGE))
    residual_baseline = bool(config.get("residual_baseline", False))
    residual_scale = np.asarray(
        config.get("residual_scale", RESIDUAL_SCALE), dtype=np.float32)
    return target, joints, release_range, residual_baseline, residual_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--ckpt", default="best_model",
                        help="checkpoint name without .zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target", type=float, nargs=2, default=None,
                        metavar=("X", "Y"))
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--cam", default="behind")
    args = parser.parse_args()

    run_name = args.run or default_name(args.target or [4.0, 0.0])
    out = Path(__file__).parent / "runs" / run_name
    (target, joints, release_range,
     residual_baseline, residual_scale) = load_config(out, args.target)

    model_path = out / args.ckpt
    if not model_path.with_suffix(".zip").exists() and args.ckpt == "best_model":
        model_path = out / "final"
    model = TD3.load(str(model_path))

    norm_path = (out / "best_vecnorm.pkl"
                 if model_path.name == "best_model" else out / "vecnorm.pkl")
    venv = DummyVecEnv([make_env(
        0, seed=123, fixed_target=target, freeze_legs=True,
        action_joints=joints, release_time_range=release_range,
        action_baseline=warmstart_action if residual_baseline else None,
        action_residual_scale=residual_scale)])
    venv = VecNormalize.load(str(norm_path), venv)
    venv.training = False
    venv.norm_reward = False

    if not args.view:
        misses = []
        landed_count = 0
        for _ in range(args.episodes):
            obs = venv.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = venv.step(action)
                if done[0]:
                    info = infos[0]
                    misses.append(float(info["objective_miss"]))
                    landed_count += int(info.get("landed") is not None)
                    break

        values = np.asarray(misses)
        print(f"target         : ({target[0]:.2f}, {target[1]:.2f}) m")
        print(f"episodes       : {args.episodes}")
        print(f"ball landed    : {landed_count}/{args.episodes}")
        print(f"mean miss      : {values.mean():.3f} m")
        print(f"median miss    : {np.median(values):.3f} m")
        print(f"within 0.50 m  : {(values < 0.50).mean() * 100:.1f}%")
        print(f"within 0.25 m  : {(values < 0.25).mean() * 100:.1f}%")
        venv.close()
        return

    raw = ThrowEnv(fixed_target=target, freeze_legs=True,
                   release_time_range=release_range, seed=123)
    obs, _ = raw.reset()
    action_ids = np.asarray([
        mujoco.mj_name2id(raw.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in joints
    ])

    viewer = mujoco.viewer.launch_passive(raw.m, raw.d)
    camera_id = mujoco.mj_name2id(
        raw.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if camera_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    while viewer.is_running():
        norm_obs = venv.normalize_obs(obs.astype(np.float32)[None])
        residual_action, _ = model.predict(norm_obs, deterministic=True)
        reduced_action = residual_action[0]
        if residual_baseline:
            reduced_action = np.clip(
                warmstart_action(raw, raw.step_i * raw.dt)
                + reduced_action * residual_scale, -1.0, 1.0)
        full_action = np.zeros(raw.n_act, dtype=np.float32)
        full_action[action_ids] = reduced_action[:-1]
        full_action[-1] = reduced_action[-1]
        obs, _, terminated, truncated, info = raw.step(full_action)

        viewer.sync()
        time.sleep(raw.dt)
        if terminated or truncated:
            print(f"target=({target[0]:.2f}, {target[1]:.2f})  "
                  f"miss={info['objective_miss']:.3f} m  "
                  f"landed={info['landed'] is not None}")
            time.sleep(1.0)
            obs, _ = raw.reset()

    viewer.close()
    venv.close()


if __name__ == "__main__":
    main()

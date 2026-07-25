"""Evaluate or view the learned walking shot-put TD3 policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from shotput_throw_env import ShotPutWalkingEnv
from shotput_throw_gym import make_shotput_env
from train_shotput_td3 import aggregate_infos


def load_run(run_dir, target_override, stage_override):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("nominal_throw_trajectory", True):
        raise ValueError(
            "this run is not a learned shot-put run")

    target = target_override or config["target"]
    walk_policy = Path(config["walk_policy"])
    if not walk_policy.is_absolute():
        walk_policy = (
            Path(__file__).resolve().parent.parent
            / walk_policy
        )
    kwargs = {
        "fixed_target": (
            None
            if config.get("random_target", False)
            else target
        ),
        "target_r": config.get("target_r", [3.5, 4.5]),
        "target_theta": config.get(
            "target_theta", [-0.10, 0.10]),
        "command": config.get("command", [1.3, 0.0, 0.0]),
        "throw_start": config.get("throw_start", 1.4),
        "prep_duration": config.get("prep_duration", 0.70),
        "release_delay": config.get(
            "release_delay", [0.12, 0.55]),
        "stage": stage_override or config.get("stage", "arm"),
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
        raise FileNotFoundError(
            f"{path}.zip does not exist. A best model is created only "
            "after deterministic improvement and shot-put style gates "
            "pass. Use --ckpt final to inspect unfinished training.")
    raise FileNotFoundError(f"checkpoint not found: {path}.zip")


def normalizer_path(run_dir, checkpoint):
    if checkpoint.name == "best_model":
        candidates = ["best_vecnorm.pkl"]
    elif checkpoint.name == "initial_model":
        candidates = ["initial_vecnorm.pkl"]
    else:
        candidates = ["vecnorm.pkl", "initial_vecnorm.pkl"]
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"normalizer not found in {run_dir}")


def build_normalizer(run_dir, checkpoint, env_kwargs):
    vec_env = DummyVecEnv([
        make_shotput_env(0, seed=123, **env_kwargs)
    ])
    vec_env = VecNormalize.load(
        str(normalizer_path(run_dir, checkpoint)),
        vec_env,
    )
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def evaluate(model, normalizer, env_kwargs, episodes):
    raw = ShotPutWalkingEnv(seed=123, **env_kwargs)
    infos = []
    for episode in range(episodes):
        observation, _ = raw.reset(seed=123 + episode)
        while True:
            normalized = normalizer.normalize_obs(
                observation[None].astype(np.float32))
            action, _ = model.predict(
                normalized,
                deterministic=True,
            )
            observation, _, terminated, truncated, info = (
                raw.step(action[0])
            )
            if terminated or truncated:
                infos.append(info)
                break
    return aggregate_infos(infos)


def print_metrics(checkpoint, target, episodes, metrics):
    print(f"checkpoint       : {checkpoint.with_suffix('.zip')}")
    print(f"target           : ({target[0]:.2f}, {target[1]:.2f}) m")
    print(f"episodes         : {episodes}")
    print(f"mean miss        : {metrics['mean_miss']:.3f} m")
    print(
        "within 0.50 m    : "
        f"{metrics['hit_rate_0_50m'] * 100:.1f}%")
    print(f"ball landed      : {metrics['landed_rate'] * 100:.1f}%")
    print(f"fall rate        : {metrics['fall_rate'] * 100:.1f}%")
    print(
        "shot-put valid   : "
        f"{metrics['style_valid_rate'] * 100:.1f}%")
    print(
        "style score      : "
        f"{metrics['mean_style_score']:.3f}")
    print(
        "ready hand error : "
        f"{metrics['mean_ready_error']:.3f} m")
    print(
        "release forward  : "
        f"{metrics['mean_release_forward_speed']:.3f} m/s")
    print(
        "release hand z   : "
        f"{metrics['mean_release_hand_height']:.3f} m")
    print(
        "release elbow    : "
        f"{metrics['mean_release_elbow_angle']:.3f} rad")
    print(
        "walk distance    : "
        f"{metrics['mean_distance_walked']:.3f} m")
    print(
        "during throw     : "
        f"{metrics['mean_distance_during_throw']:.3f} m")
    moving_fraction = metrics[
        "mean_positive_velocity_fraction_during_throw"]
    print(
        "moving in throw  : "
        f"{moving_fraction * 100:.1f}%")
    print(
        "max line error   : "
        f"{metrics['mean_max_abs_lateral_error']:.3f} m")
    print(
        "selection score  : "
        f"{metrics['selection_score']:.3f} (lower is better)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="run folder name under train/runs",
    )
    parser.add_argument("--ckpt", default="best_model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--target",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
    )
    parser.add_argument(
        "--stage",
        choices=["arm", "upper", "full"],
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--cam", default="follow")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    _, target, env_kwargs = load_run(
        run_dir,
        args.target,
        args.stage,
    )
    checkpoint = find_checkpoint(run_dir, args.ckpt)
    model = TD3.load(str(checkpoint), device=args.device)
    normalizer = build_normalizer(
        run_dir,
        checkpoint,
        env_kwargs,
    )

    if not args.view:
        metrics = evaluate(
            model,
            normalizer,
            env_kwargs,
            args.episodes,
        )
        print_metrics(
            checkpoint,
            target,
            args.episodes,
            metrics,
        )

        initial_path = run_dir / "initial_metrics.json"
        proof_path = run_dir / "learning_proof.json"
        if initial_path.exists():
            initial = json.loads(
                initial_path.read_text(encoding="utf-8"))
            print("")
            print("learning audit")
            print("--------------")
            print("nominal throw    : NONE")
            print(
                "initial score    : "
                f"{initial['selection_score']:.3f}")
            score_improvement = (
                initial["selection_score"]
                - metrics["selection_score"]
            )
            print(
                "score improvement: "
                f"{score_improvement:.3f}")
        if proof_path.exists():
            proof = json.loads(
                proof_path.read_text(encoding="utf-8"))
            final = proof.get("final", {})
            print(
                "actor L2 change  : "
                f"{final.get('actor_l2_delta', float('nan')):.6f}")
            print(
                "actor changed    : "
                f"{proof.get('actor_changed', False)}")
            print(
                "proven best      : "
                f"{proof.get('best_model_exists', False)}")
        normalizer.close()
        return

    raw = ShotPutWalkingEnv(seed=123, **env_kwargs)
    observation, _ = raw.reset()
    with mujoco.viewer.launch_passive(raw.m, raw.d) as viewer:
        camera_id = mujoco.mj_name2id(
            raw.m,
            mujoco.mjtObj.mjOBJ_CAMERA,
            args.cam,
        )
        if camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id

        while viewer.is_running():
            normalized = normalizer.normalize_obs(
                observation[None].astype(np.float32))
            action, _ = model.predict(
                normalized,
                deterministic=True,
            )
            observation, _, terminated, truncated, info = (
                raw.step(action[0])
            )
            viewer.sync()
            time.sleep(raw.dt)
            if terminated or truncated:
                print(
                    f"miss={info['objective_miss']:.3f} m, "
                    f"style={info['shotput_style_score']:.3f}, "
                    f"valid={info['shotput_style_valid']}, "
                    f"release_vx={info['release_forward_speed']:.3f}, "
                    f"walked={info['distance_walked']:.3f} m, "
                    f"fallen={info['fallen']}"
                )
                time.sleep(0.7)
                observation, _ = raw.reset()
    normalizer.close()


if __name__ == "__main__":
    main()

"""
Evaluate a trained policy, with an optional viewer.

    python eval_policy.py --run throw_v1 --episodes 50
    python eval_policy.py --run throw_v1 --view
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_env import make_env
from env import ThrowEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="throw_v1")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--cam", default="behind")
    ap.add_argument("--target", type=float, nargs=2, default=None,
                    help="fixed target x y")
    args = ap.parse_args()

    out = Path(__file__).parent / "runs" / args.run
    model = PPO.load(str(out / args.ckpt))

    kw = {}
    if args.target:
        kw["fixed_target"] = args.target

    # rebuild the observation normalisation used during training
    venv = DummyVecEnv([make_env(0, seed=123, **kw)])
    venv = VecNormalize.load(str(out / "vecnorm.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    if not args.view:
        misses, no_release = [], 0
        for _ in range(args.episodes):
            obs = venv.reset()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = venv.step(action)
                if done[0]:
                    info = infos[0]
                    if info.get("miss") is not None:
                        misses.append(info["miss"])
                    else:
                        no_release += 1
                    break
        m = np.array(misses)
        print(f"episodes      : {args.episodes}")
        print(f"no release    : {no_release}")
        print(f"mean miss     : {m.mean():.2f} m")
        print(f"median miss   : {np.median(m):.2f} m")
        print(f"within 1.0 m  : {(m < 1.0).mean()*100:.0f}%")
        print(f"within 0.5 m  : {(m < 0.5).mean()*100:.0f}%")
        print(f"within 0.25 m : {(m < 0.25).mean()*100:.0f}%")
        return

    # --- viewer mode: drive the raw env so we can render -------------
    raw = ThrowEnv(**kw, seed=123)
    obs, _ = raw.reset()
    viewer = mujoco.viewer.launch_passive(raw.m, raw.d)
    cid = mujoco.mj_name2id(raw.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cid >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cid

    while viewer.is_running():
        norm = venv.normalize_obs(obs.astype(np.float32)[None])
        action, _ = model.predict(norm, deterministic=True)
        obs, r, term, trunc, info = raw.step(action[0])

        viewer.sync()
        time.sleep(raw.dt)

        if term or trunc:
            if info.get("miss") is not None:
                print(f"target=({info['target'][0]:5.2f},{info['target'][1]:5.2f})  "
                      f"miss={info['miss']:.2f} m  "
                      f"hand={info['hand_speed']:.2f} m/s")
            else:
                print("no release")
            time.sleep(1.0)
            obs, _ = raw.reset()

    viewer.close()


if __name__ == "__main__":
    main()
"""
Drive ThrowEnv with the hand-tuned swing from check_throw.py.

If this reproduces the ~0.45 m result, the env wrapper is faithful and any
later failure is in the learning code, not the physics.
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from env import ThrowEnv

# same numbers as check_throw.py
WINDUP = {"right_shoulder_pitch_joint":  1.8,
          "right_shoulder_roll_joint":  -0.2,
          "right_elbow_joint":           1.9}
RELEASE = {"right_shoulder_pitch_joint": -2.0,
           "right_shoulder_roll_joint":  -0.2,
           "right_elbow_joint":           0.1}
T_WINDUP, T_SWING, T_LET_GO = 0.60, 0.85, 0.783


def smooth(a, b, t):
    t = np.clip(t, 0.0, 1.0)
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * t))


def scripted_action(env, t, letgo):
    """Convert the absolute-angle swing into the env's residual action."""
    m = env.m
    a = np.zeros(env.n_act)

    for name, wind in WINDUP.items():
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if t < T_WINDUP:
            ang = smooth(env.q_stand[i], wind, t / T_WINDUP)
        else:
            ang = smooth(wind, RELEASE[name],
                         (t - T_WINDUP) / (T_SWING - T_WINDUP))
        a[i] = np.clip((ang - env.q_stand[i]) / env.scale[i], -1.0, 1.0)

    a[m.nu] = np.clip((t + env.dt - letgo) / env.dt, -1.0, 1.0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letgo", type=float, default=T_LET_GO)
    ap.add_argument("--random-target", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--cam", default="behind")
    args = ap.parse_args()

    env = ThrowEnv(fixed_target=None if args.random_target else [4.0, 0.0],
                   freeze_legs=True, seed=0)

    print(f"obs dim = {env.n_obs}   act dim = {env.n_act}")
    print(f"control rate = {1/env.dt:.0f} Hz   sim dt = {env.m.opt.timestep}")
    print(f"max steps = {env.max_steps}\n")

    if args.sweep:
        for lg in np.arange(0.776, 0.786, 0.001):
            env.reset()
            t = 0.0
            while True:
                _, _, term, trunc, info = env.step(scripted_action(env, t, lg))
                t += env.dt
                if term or trunc:
                    break
            miss = info["miss"]
            print(f"letgo={lg:.3f}  miss={miss:.2f} m" if miss is not None
                  else f"letgo={lg:.3f}  no landing")
        return

    viewer = None
    if args.view:
        viewer = mujoco.viewer.launch_passive(env.m, env.d)
        cid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
        if cid >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cid

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total, t = 0.0, 0.0
        while True:
            obs, r, term, trunc, info = env.step(
                scripted_action(env, t, args.letgo))
            total += r
            t += env.dt
            if viewer:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(env.dt)
            if term or trunc:
                break

        tgt = info["target"]
        if info["landed"] is not None:
            print(f"ep{ep}  target=({tgt[0]:5.2f},{tgt[1]:5.2f})  "
                  f"landed=({info['landed'][0]:5.2f},{info['landed'][1]:5.2f})  "
                  f"miss={info['miss']:5.2f} m  "
                  f"hand={info['hand_speed']:5.2f} m/s  "
                  f"return={total:7.2f}")
        else:
            print(f"ep{ep}  no landing (released={info['released']})  "
                  f"return={total:7.2f}")

        if viewer:
            time.sleep(0.8)          # pause between throws so you can see it
            if not viewer.is_running():
                break

    if viewer:
        viewer.close()


if __name__ == "__main__":
    main()
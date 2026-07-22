import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# ---- TUNE THESE ----------------------------------------------------
WINDUP = {"right_shoulder_pitch_joint":  1.8,   # arm BACK (positive)
          "right_shoulder_roll_joint":  -0.2,
          "right_elbow_joint":           1.9}   # elbow bent
RELEASE = {"right_shoulder_pitch_joint": -2.0,  # arm FORWARD and up (negative)
           "right_shoulder_roll_joint":  -0.2,
           "right_elbow_joint":           0.1}  # elbow extended

T_WINDUP = 0.60
T_SWING  = 0.85
T_LET_GO = 0.77
OFFSET   = np.array([0.145, 0.0, 0.0])   # must match relpose in the XML
# --------------------------------------------------------------------

path = Path(__file__).parent.parent / "models" / "throw_scene.xml"
m = mujoco.MjModel.from_xml_path(str(path))

hand   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
grip   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
bj     = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
ball_q = m.jnt_qposadr[bj]
ball_v = m.jnt_dofadr[bj]
target = m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")]


def aid(name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def smooth(a, b, t):
    """cosine ease, so we never command a step change"""
    t = np.clip(t, 0.0, 1.0)
    return a + (b - a) * 0.5 * (1 - np.cos(np.pi * t))


def body_vel(d, bid):
    """world-frame linear velocity of a body - exact, not finite-differenced"""
    v6 = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
    return v6[3:6]          # [0:3] is angular, [3:6] is linear


def run(letgo, swing=T_SWING, view=False, cam=None, verbose=False):
    """One throw. Returns (hand_speed, launch_velocity_vec, landing_xyz_or_None)."""
    d = mujoco.MjData(m)
    state = {}

    def reset():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        R = d.xmat[hand].reshape(3, 3)
        d.qpos[ball_q:ball_q+3]   = d.xpos[hand] + R @ OFFSET
        d.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
        d.eq_active[grip] = 1                    # re-close the grip
        mujoco.mj_forward(m, d)                  # refresh after moving the ball
        state.update(released=False, hand_speed=0.0,
                     vel=np.zeros(3), landed=None)

    reset()
    q_stand = d.qpos[7:7+m.nu].copy()
    start   = {k: q_stand[aid(k)] for k in WINDUP}

    viewer = None
    if view:
        viewer = mujoco.viewer.launch_passive(m, d)
        if cam is not None:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cid < 0:
                print(f"warning: no camera named '{cam}'")
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cid

    prev_time = d.time

    while True:
        if d.time < prev_time:               # viewer reset: time jumped back
            reset()

        if not view and d.time >= 4.0:
            break
        if view and not viewer.is_running():
            break

        # --- command the swing --------------------------------------
        d.ctrl[:] = q_stand
        for k in WINDUP:
            if d.time < T_WINDUP:
                ang = smooth(start[k], WINDUP[k], d.time / T_WINDUP)
            else:
                ang = smooth(WINDUP[k], RELEASE[k],
                             (d.time - T_WINDUP) / (swing - T_WINDUP))
            d.ctrl[aid(k)] = ang

        # --- release -------------------------------------------------
        capture = False
        if not state["released"] and d.time >= letgo:
            state["hand_speed"] = np.linalg.norm(body_vel(d, hand))
            d.eq_active[grip] = 0
            state["released"] = True
            capture = True

        prev_time = d.time
        mujoco.mj_step(m, d)

        if capture:                          # ball is free now
            state["vel"] = d.qvel[ball_v:ball_v+3].copy()

        if state["released"] and state["landed"] is None and d.qpos[ball_q+2] < 0.025:
            state["landed"] = d.qpos[ball_q:ball_q+3].copy()
            if verbose:
                dist = np.linalg.norm(state["landed"][:2] - target[:2])
                print(f"landed at {state['landed'][:2]}  ->  {dist:.2f} m from target")

        if viewer:
            viewer.sync()
            time.sleep(m.opt.timestep)

    if viewer:
        viewer.close()
    return state["hand_speed"], state["vel"], state["landed"]


ap = argparse.ArgumentParser()
ap.add_argument("--view",  action="store_true")
ap.add_argument("--cam",   default="behind", help="side | behind | free")
ap.add_argument("--sweep", action="store_true", help="coarse release-time sweep")
ap.add_argument("--fine",  action="store_true", help="fine sweep near the optimum")
ap.add_argument("--letgo", type=float, default=T_LET_GO)
ap.add_argument("--swing", type=float, default=T_SWING,
                help="lower = faster commanded swing; use to test the kp ceiling")
args = ap.parse_args()

if args.sweep or args.fine:
    rng = np.arange(0.765, 0.795, 0.002) if args.fine else np.arange(0.55, 0.92, 0.02)
    print(f"T_SWING = {args.swing}")
    print(f"{'letgo':>6} {'hand':>6} {'vx':>7} {'vy':>7} {'vz':>7} {'dist':>7}")
    for t in rng:
        hs, vel, land = run(float(t), swing=args.swing)
        dist = np.linalg.norm(land[:2] - target[:2]) if land is not None else -1.0
        print(f"{t:6.3f} {hs:6.2f} {vel[0]:7.2f} {vel[1]:7.2f} {vel[2]:7.2f} {dist:7.2f}")
else:
    cam = None if args.cam == "free" else args.cam
    hs, vel, land = run(args.letgo, swing=args.swing, view=args.view,
                        cam=cam, verbose=args.view)
    print(f"hand speed at release : {hs:6.2f} m/s")
    print(f"launch velocity       : {np.linalg.norm(vel):6.2f} m/s  {vel}")
    if land is not None:
        print(f"landed at             : {land[:2]}")
        print(f"distance to target    : {np.linalg.norm(land[:2] - target[:2]):6.2f} m")
    else:
        print("ball never landed within 4s")
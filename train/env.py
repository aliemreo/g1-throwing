"""
Whole-body dynamic throwing environment for the Unitree G1 in MuJoCo.

Gymnasium-style interface, but with no gymnasium dependency so it stays easy
to debug. Wrap it later for SB3 / Brax.

Design notes:
  - Actions are RESIDUALS around the 'stand' keyframe, so a zero action means
    "hold the standing pose". Random init therefore produces near-sane motion.
  - Action vector is length 30: 29 joint offsets + 1 release control.
  - By default release fires on a zero-crossing of action[29], interpolated to
    sub-step precision. For TD3, release_time_range maps the initial continuous
    release action to an absolute release time, which is substantially easier
    for a deterministic policy to optimise than a discontinuous zero-crossing.
  - Reward contains exactly one objective: landing accuracy. During flight it
    is expressed as improvement in predicted landing error; at episode end the
    potential is replaced by the actual landing/current ball error. The sum of
    undiscounted rewards is therefore initial_error - final_error. There are no
    pose, energy, speed, survival, release, or terminal-bonus terms.
"""

import numpy as np
import mujoco
from pathlib import Path

G = 9.81

SCENE = Path(__file__).parent.parent / "models" / "throw_scene.xml"

# Must match the 'grip' weld relpose in throw_scene.xml
OFFSET = np.array([0.145, 0.0, 0.0])

# Joints whose names contain these get the given residual scale (radians).
# The throwing arm needs a wide range - the hand-tuned swing spans ~3.8 rad -
# while legs and waist only need small corrections.
SCALE_RULES = [
    ("right_shoulder", 2.0),
    ("right_elbow",    2.0),
    ("right_wrist",    1.5),
    ("waist",          0.4),
    ("left_",          0.3),
    ("hip",            0.3),
    ("knee",           0.3),
    ("ankle",          0.3),
]
DEFAULT_SCALE = 0.5


class ThrowEnv:
    def __init__(self,
                 decimation=5,
                 max_seconds=3.0,
                 target_r=(2.5, 6.0),
                 target_theta=(-0.5, 0.5),
                 fixed_target=None,
                 freeze_legs=True,
                 release_time_range=None,
                 elbow_min=None,
                 seed=None):
        """
        target_r / target_theta : sampling ranges. Narrow them for curriculum.
        fixed_target            : [x, y] to disable randomisation (testing).
        freeze_legs             : zero out leg actions. The pelvis is pinned,
                                  so leg motion is currently useless and only
                                  enlarges the exploration problem.
        """
        assert SCENE.exists(), f"scene not found: {SCENE}"
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)

        # Human-like elbow: the model allows the elbow to hyperextend to
        # -1.047 rad (60 deg past straight), which no person can do and which RL
        # exploits for a stronger throw. Raise the lower limit (0 rad = straight
        # arm) so the elbow can flex and straighten but not hyperextend. Applied
        # per-instance (this ThrowEnv only); the walk+throw env is unaffected.
        if elbow_min is not None:
            aid = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_joint")
            jid = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint")
            if aid >= 0:
                self.m.actuator_ctrlrange[aid, 0] = max(
                    self.m.actuator_ctrlrange[aid, 0], float(elbow_min))
            if jid >= 0:
                self.m.jnt_range[jid, 0] = max(
                    self.m.jnt_range[jid, 0], float(elbow_min))

        self.decimation = decimation
        self.dt = self.m.opt.timestep * decimation
        self.max_steps = int(max_seconds / self.dt)
        self.target_r = target_r
        self.target_theta = target_theta
        self.fixed_target = fixed_target
        self.release_time_range = release_time_range
        if release_time_range is not None:
            if (len(release_time_range) != 2
                    or release_time_range[0] >= release_time_range[1]):
                raise ValueError("release_time_range must be (minimum, maximum)")

        m = self.m
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                      "right_wrist_yaw_link")
        self.grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
        self.tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_q = m.jnt_qposadr[bj]
        self.ball_v = m.jnt_dofadr[bj]

        # residual scale, per actuator
        self.scale = np.full(m.nu, DEFAULT_SCALE)
        for i in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for pat, s in SCALE_RULES:
                if pat in name:
                    self.scale[i] = s
                    break

        self.leg_mask = np.ones(m.nu)
        if freeze_legs:
            for i in range(m.nu):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if any(k in name for k in ("hip", "knee", "ankle")):
                    self.leg_mask[i] = 0.0

        # standing pose, read once from the keyframe
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        mujoco.mj_forward(m, self.d)
        self.q_stand = self.d.qpos[7:7 + m.nu].copy()

        self.n_act = m.nu + 1          # +1 for the release signal
        obs, _ = self.reset()          # also establishes target/step_i/etc
        self.n_obs = len(obs)

    # ---------------------------------------------------------------- utils

    def _ball_pos(self):
        return self.d.qpos[self.ball_q:self.ball_q + 3].copy()

    def _ball_vel(self):
        return self.d.qvel[self.ball_v:self.ball_v + 3].copy()

    def _hand_vel(self):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY,
                                 self.hand, v6, 0)
        return v6[3:6].copy()

    def predicted_landing(self):
        """Where the ball would land if released right now (ballistic, no drag)."""
        p, v = self._ball_pos(), self._ball_vel()
        z = max(p[2], 1e-6)
        disc = v[2] ** 2 + 2 * G * z
        t = (v[2] + np.sqrt(disc)) / G
        return p[:2] + v[:2] * t

    def _landing_potential(self):
        """Negative target error if the ball were released at this instant."""
        error = np.linalg.norm(self.predicted_landing() - self.target[:2])
        return -float(error)

    def _actual_potential(self):
        """Negative target error of the ball's current/landed XY position."""
        pos = self.landed if self.landed is not None else self._ball_pos()
        return -float(np.linalg.norm(pos[:2] - self.target[:2]))

    def _obs(self):
        d, m = self.d, self.m
        hand_pos = d.xpos[self.hand]
        return np.concatenate([
            d.qpos[7:7 + m.nu] - self.q_stand,   # joint offsets from stand
            d.qvel[6:6 + m.nu] * 0.1,            # joint velocities, scaled
            self._ball_pos() - hand_pos,         # ball relative to hand
            self._ball_vel() * 0.1,
            self.target[:2],                     # where to throw
            [float(self.released)],
            [self.step_i / self.max_steps],      # phase through the episode
        ]).astype(np.float64)

    # ---------------------------------------------------------------- api

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.m, self.d

        # --- target ---------------------------------------------------
        if self.fixed_target is not None:
            tx, ty = self.fixed_target
        else:
            r = self.rng.uniform(*self.target_r)
            th = self.rng.uniform(*self.target_theta)
            tx, ty = r * np.cos(th), r * np.sin(th)
        m.site_pos[self.tid] = [tx, ty, 0.01]
        self.target = np.array([tx, ty, 0.01])

        # --- state ----------------------------------------------------
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)

        R = d.xmat[self.hand].reshape(3, 3)
        d.qpos[self.ball_q:self.ball_q + 3] = d.xpos[self.hand] + R @ OFFSET
        d.qpos[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
        d.eq_active[self.grip] = 1
        mujoco.mj_forward(m, d)          # refresh before the solver runs

        self.released = False
        self.landed = None
        self.launch_vel = np.zeros(3)
        self.hand_speed = 0.0
        self.step_i = 0
        self.prev_signal = -1.0
        self.scheduled_release_time = None
        self.phi_prev = self._landing_potential()

        return self._obs(), {}

    def step(self, action):
        """action: length nu+1. [:nu] joint residuals in [-1,1], [nu] release."""
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        m, d = self.m, self.d

        # --- residual joint targets -----------------------------------
        ctrl = self.q_stand + action[:m.nu] * self.scale * self.leg_mask
        d.ctrl[:] = np.clip(ctrl, m.actuator_ctrlrange[:, 0],
                            m.actuator_ctrlrange[:, 1])

        # --- release, resolved to a simulator sub-step ----------------
        signal = action[m.nu]
        release_at = None
        if self.release_time_range is not None:
            if self.scheduled_release_time is None:
                lo, hi = self.release_time_range
                self.scheduled_release_time = (
                    lo + (signal + 1.0) * 0.5 * (hi - lo))
            if not self.released:
                until_release = self.scheduled_release_time - d.time
                horizon = self.decimation * m.opt.timestep
                if until_release < horizon:
                    release_at = int(np.clip(
                        np.ceil(max(until_release - 1e-9, 0.0)
                                / m.opt.timestep),
                        0, self.decimation - 1))
        elif not self.released and self.prev_signal < 0 <= signal:
            frac = -self.prev_signal / (signal - self.prev_signal + 1e-9)
            release_at = int(np.clip(frac * self.decimation,
                                     0, self.decimation - 1))
        self.prev_signal = signal

        # --- advance physics ------------------------------------------
        for k in range(self.decimation):
            if release_at is not None and k == release_at:
                self.hand_speed = np.linalg.norm(self._hand_vel())
                d.eq_active[self.grip] = 0
                self.released = True
                mujoco.mj_step(m, d)
                self.launch_vel = self._ball_vel()
                continue
            mujoco.mj_step(m, d)

            if self.released and self.landed is None and \
                    d.qpos[self.ball_q + 2] < 0.025:
                self.landed = self._ball_pos()
                break

        self.step_i += 1

        terminated = self.landed is not None
        truncated = self.step_i >= self.max_steps

        # This is the only reward signal: reduction in target error. Use the
        # ballistic landing prediction while acting, then the actual ball
        # position at the end. With gamma=1 the episode return telescopes to
        # initial_error - final_error, so no auxiliary objective is hidden in
        # the shaping.
        phi = (self._actual_potential() if terminated or truncated
               else self._landing_potential())
        reward = phi - self.phi_prev
        self.phi_prev = phi

        landing_miss = (float(np.linalg.norm(
            self.landed[:2] - self.target[:2]))
            if self.landed is not None else None)
        objective_miss = -phi if terminated or truncated else None

        info = {"landed": self.landed,
                "released": self.released,
                "hand_speed": self.hand_speed,
                "launch_vel": self.launch_vel,
                "scheduled_release_time": self.scheduled_release_time,
                "target": self.target[:2].copy(),
                "miss": landing_miss,
                "objective_miss": objective_miss}

        return self._obs(), reward, terminated, truncated, info

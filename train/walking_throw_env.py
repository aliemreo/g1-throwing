"""MuJoCo Level-3 environment: walk forward while throwing at a target.

The lower body starts from a pretrained recurrent walking policy. TD3 learns
small leg residuals, upper-body balance/throw residuals, and release timing.
Unlike the fixed-pelvis task this environment needs locomotion and balance
reward terms; landing accuracy remains the throwing objective. The default
timeline contains a full walking lead-in, a throw while forward locomotion is
already active, and a walking follow-through.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

from walk_policy import (
    DEFAULT_ANGLES,
    DEFAULT_POLICY,
    GAIT_PERIOD,
    LEG_JOINTS,
    WalkPolicy,
    configure_leg_pd,
    joint_addresses,
    projected_gravity,
)


G = 9.81
SCENE = (
    Path(__file__).resolve().parent.parent
    / "models" / "walk_throw_scene.xml"
)
BALL_OFFSET = np.asarray([0.145, 0.0, 0.0], dtype=np.float64)

WAIST_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
)
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

WINDUP = {
    "right_shoulder_pitch_joint": 1.735366,
    "right_shoulder_roll_joint": -0.711473,
    "right_elbow_joint": 1.913143,
}
RELEASE_POSE = {
    "right_shoulder_pitch_joint": -1.795524,
    "right_shoulder_roll_joint": -0.625955,
    "right_elbow_joint": 0.073054,
}
T_WINDUP = 0.600418
T_SWING = 0.889800
# Re-tuned after composing the throw with the 0.8 s walking gait. The pinned
# controller's 0.784 s release is too early once the pelvis is dynamic.
T_LET_GO = 0.7775

# Terminal accuracy reward. The dense potential shaping alone left "throw short
# and stay upright" a better-scoring policy than "aim well", so RL collapsed to
# landing ~2 m regardless of the target. A large end-of-episode bonus for landing
# near the target makes accurate aiming clearly worth more than the marginal
# balance gain of a weak throw (peak 25 at a perfect hit, ~9 at 0.5 m, ~0 by 2 m).
LANDING_BONUS = 25.0
LANDING_BONUS_SCALE = 0.5

LEG_RESIDUAL_SCALE = 0.06
WAIST_RESIDUAL_SCALE = np.asarray([0.20, 0.16, 0.16])
LEFT_ARM_RESIDUAL_SCALE = np.asarray([0.30, 0.24, 0.24, 0.30])
# Calibration of the pinned-base throw after composing it with the walking
# gait. This zero-policy baseline lands within roughly 0.21 m of (4, 0).
RIGHT_ARM_NOMINAL_OFFSET = np.asarray([-0.075, 0.040, 0.150, 0.0])
RIGHT_ARM_RESIDUAL_SCALE = np.asarray([0.03, 0.03, 0.04, 0.03])

ACTION_GROUPS = {
    "leg_residual": (0, 12),
    "waist_residual": (12, 15),
    "left_arm_residual": (15, 19),
    "right_arm_residual": (19, 23),
    "release_time": (23, 24),
}
N_ACTIONS = 24


def _smooth(start, end, phase):
    phase = np.clip(phase, 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * phase))
    return start + (end - start) * blend


def _yaw_from_quat(quat):
    """World yaw for a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = quat
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class WalkingThrowEnv:
    """Dependency-light environment with the Gymnasium step signature."""

    def __init__(
        self,
        max_seconds=5.0,
        command=(1.3, 0.0, 0.0),
        fixed_target=(4.0, 0.0),
        target_r=(3.0, 5.0),
        target_theta=(-0.25, 0.25),
        throw_start=1.4,
        release_window=0.006,
        release_center_offset=T_LET_GO,
        line_correction=True,
        line_position_gain=4.0,
        line_heading_gain=1.5,
        stage="throw",
        walk_policy_path=DEFAULT_POLICY,
        walk_policy_device="cpu",
        seed=None,
    ):
        if not SCENE.exists():
            raise FileNotFoundError(f"scene not found: {SCENE}")
        self.m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)

        # The pretrained policy was deployed at 50 Hz over 0.002 s physics.
        self.m.opt.timestep = 0.002
        self.decimation = 10
        self.dt = self.m.opt.timestep * self.decimation
        self.max_steps = int(float(max_seconds) / self.dt)
        self.command = np.asarray(command, dtype=np.float64)
        if self.command.shape != (3,):
            raise ValueError("command must be [vx, vy, yaw_rate]")
        self.fixed_target = (
            None if fixed_target is None
            else np.asarray(fixed_target, dtype=np.float64)
        )
        self.target_r = tuple(target_r)
        self.target_theta = tuple(target_theta)
        self.throw_start = float(throw_start)
        # The release time the zero action maps to. Defaults to the nominal
        # T_LET_GO swing release. For the aiming task it is re-centred on the
        # middle of the useful 3-5 m release band so action[23] in [-1, 1] maps
        # smoothly onto that whole band and clipping keeps every release inside
        # it - instead of most of the action range hitting the catastrophic
        # early-release / cliff regions that poison the critic.
        self.nominal_release_time = self.throw_start + float(
            release_center_offset)
        self.release_window = float(release_window)
        self.line_correction = bool(line_correction)
        self.line_position_gain = float(line_position_gain)
        self.line_heading_gain = float(line_heading_gain)
        self.walk_command = self.command.copy()
        if stage not in {"release", "throw", "full"}:
            raise ValueError("stage must be 'release', 'throw', or 'full'")
        self.stage = stage

        self.leg_actuator_ids = configure_leg_pd(self.m)
        self.leg_qpos_ids, self.leg_qvel_ids = joint_addresses(
            self.m, LEG_JOINTS)
        self.waist_actuator_ids = self._actuator_ids(WAIST_JOINTS)
        self.left_arm_actuator_ids = self._actuator_ids(LEFT_ARM_JOINTS)
        self.right_arm_actuator_ids = self._actuator_ids(RIGHT_ARM_JOINTS)
        actuator_joint_names = tuple(
            mujoco.mj_id2name(
                self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.m.nu)
        )
        self.all_joint_qpos, self.all_joint_qvel = joint_addresses(
            self.m, actuator_joint_names)

        self.walk = WalkPolicy(
            self.m, walk_policy_path, device=walk_policy_device)
        self.hand = self._body_id("right_wrist_yaw_link")
        self.grip = self._equality_id("grip")
        self.target_site = self._site_id("target")
        ball_joint = self._joint_id("ball_free")
        self.ball_qpos = self.m.jnt_qposadr[ball_joint]
        self.ball_qvel = self.m.jnt_dofadr[ball_joint]
        # A ball resting on the plane has its centre at z = radius, so derive the
        # landing z-threshold from the actual ball geom size instead of hardcoding
        # it (0.023 assumed a 2 cm ball; a 4 cm ball rests at z = 0.04 and would
        # never be flagged as landed under the old constant).
        ball_geom = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ball_radius = float(self.m.geom_size[ball_geom][0])
        self.landing_z = self.ball_radius + 0.003

        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.q_nominal = self.d.qpos[self.all_joint_qpos].copy()
        self.q_nominal[:12] = DEFAULT_ANGLES

        self.n_act = N_ACTIONS
        obs, _ = self.reset()
        self.n_obs = int(obs.size)

    # -------------------------------------------------------------- ids

    def _actuator_ids(self, names):
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx < 0:
                raise ValueError(f"actuator not found: {name}")
            ids.append(idx)
        return np.asarray(ids, dtype=np.int32)

    def _body_id(self, name):
        idx = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"body not found: {name}")
        return idx

    def _joint_id(self, name):
        idx = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if idx < 0:
            raise ValueError(f"joint not found: {name}")
        return idx

    def _site_id(self, name):
        idx = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"site not found: {name}")
        return idx

    def _equality_id(self, name):
        idx = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if idx < 0:
            raise ValueError(f"equality not found: {name}")
        return idx

    # ------------------------------------------------------------ state

    def _ball_pos(self):
        return self.d.qpos[self.ball_qpos:self.ball_qpos + 3].copy()

    def _ball_vel(self):
        return self.d.qvel[self.ball_qvel:self.ball_qvel + 3].copy()

    def _hand_velocity(self):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.m, self.d, mujoco.mjtObj.mjOBJ_BODY,
            self.hand, velocity, 0)
        return velocity[3:6].copy()

    def _predicted_landing(self):
        position = self._ball_pos()
        velocity = self._ball_vel()
        z = max(position[2], 1e-6)
        flight = (
            velocity[2]
            + np.sqrt(velocity[2] ** 2 + 2.0 * G * z)
        ) / G
        return position[:2] + velocity[:2] * flight

    def _throw_potential(self):
        xy = (
            self.landed[:2]
            if self.landed is not None
            else self._predicted_landing()
        )
        return -float(np.linalg.norm(xy - self.target[:2]))

    def _fall_state(self):
        gravity = projected_gravity(self.d.qpos[3:7])
        fallen = bool(
            self.d.qpos[2] < 0.52
            or gravity[2] > -0.45
            or not np.isfinite(self.d.qpos).all()
        )
        return fallen, gravity

    def _upper_body_baseline(self, time_s):
        target = self.q_nominal.copy()
        local_time = time_s - self.throw_start
        for name in RIGHT_ARM_JOINTS:
            aid = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if name not in WINDUP:
                continue
            start = self.q_nominal[aid]
            if local_time <= 0.0:
                angle = start
            elif local_time < T_WINDUP:
                angle = _smooth(
                    start, WINDUP[name], local_time / T_WINDUP)
            else:
                angle = _smooth(
                    WINDUP[name], RELEASE_POSE[name],
                    (local_time - T_WINDUP) / (T_SWING - T_WINDUP))
            target[aid] = angle
        return target

    def _line_state(self):
        """Cross-track and heading errors relative to the reset heading."""
        position_delta = self.d.qpos[:2] - self.start_xy
        lateral_axis = np.asarray(
            [-np.sin(self.start_yaw), np.cos(self.start_yaw)])
        forward_axis = np.asarray(
            [np.cos(self.start_yaw), np.sin(self.start_yaw)])
        lateral_error = float(np.dot(lateral_axis, position_delta))
        forward_distance = float(np.dot(forward_axis, position_delta))
        lateral_velocity = float(np.dot(lateral_axis, self.d.qvel[:2]))
        forward_velocity = float(np.dot(forward_axis, self.d.qvel[:2]))
        heading_error = float(_wrap_angle(
            _yaw_from_quat(self.d.qpos[3:7]) - self.start_yaw))
        return (
            forward_distance,
            lateral_error,
            forward_velocity,
            lateral_velocity,
            heading_error,
        )

    def _corrected_walk_command(self):
        command = self.command.copy()
        if not self.line_correction:
            return command
        _, lateral_error, _, lateral_velocity, heading_error = (
            self._line_state())
        command[1] = np.clip(
            self.command[1]
            - self.line_position_gain * lateral_error
            - 0.2 * lateral_velocity,
            -0.8,
            0.8,
        )
        command[2] = np.clip(
            self.command[2]
            - self.line_heading_gain * heading_error
            - 0.5 * lateral_error
            - 0.2 * self.d.qvel[5],
            -1.0,
            1.0,
        )
        return command

    def _capture_phase_metrics(self):
        time_s = float(self.d.time)
        forward_distance, lateral_error, forward_velocity, _, _ = (
            self._line_state())
        self.max_abs_lateral_error = max(
            self.max_abs_lateral_error, abs(lateral_error))
        if (
            self.forward_at_throw_start is None
            and time_s >= self.throw_start - 1e-9
        ):
            self.forward_at_throw_start = forward_distance
        if (
            self.forward_at_throw_end is None
            and time_s >= self.throw_start + T_SWING - 1e-9
        ):
            self.forward_at_throw_end = forward_distance
        if self.throw_start <= time_s <= self.throw_start + T_SWING:
            self.throw_velocity_samples += 1
            self.throw_positive_velocity_samples += int(
                forward_velocity > 0.02)
            self.min_throw_forward_velocity = min(
                self.min_throw_forward_velocity, forward_velocity)

    def _observation(self):
        _, gravity = self._fall_state()
        joints = self.d.qpos[self.all_joint_qpos]
        velocities = self.d.qvel[self.all_joint_qvel]
        base_position = self.d.qpos[:3]
        phase = (self.d.time % GAIT_PERIOD) / GAIT_PERIOD
        # Dedicated aim signal: the target distance from the origin, constant for
        # the whole episode and centred near the 4 m nominal. Unlike
        # target - base_position (whose variance is dominated by walk progress,
        # burying the cross-target signal), this is a clean, walk-independent
        # feature so the policy can key its release timing on the target. It is
        # repeated so the feature is salient among the ~90 other observations.
        target_distance = float(np.linalg.norm(self.target[:2]))
        aim = (target_distance - 4.0)
        return np.concatenate([
            self.d.qvel[:3] * 0.5,
            self.d.qvel[3:6] * 0.25,
            gravity,
            joints - self.q_nominal,
            velocities * 0.05,
            self._ball_pos() - self.d.xpos[self.hand],
            self._ball_vel() * 0.1,
            self.target[:2] - base_position[:2],
            [aim, aim, aim, aim],
            self.walk_command,
            self.walk.previous_action,
            [float(self.released)],
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
            [self.step_index / self.max_steps],
        ]).astype(np.float32)

    # --------------------------------------------------------------- api

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.fixed_target is not None:
            tx, ty = self.fixed_target
        else:
            radius = self.rng.uniform(*self.target_r)
            theta = self.rng.uniform(*self.target_theta)
            tx, ty = radius * np.cos(theta), radius * np.sin(theta)
        self.target = np.asarray([tx, ty, 0.01], dtype=np.float64)
        self.m.site_pos[self.target_site] = self.target

        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.d.qpos[2] = 0.80
        self.d.qpos[self.leg_qpos_ids] = DEFAULT_ANGLES
        self.d.ctrl[:] = self.q_nominal
        self.d.ctrl[self.leg_actuator_ids] = DEFAULT_ANGLES
        mujoco.mj_forward(self.m, self.d)

        rotation = self.d.xmat[self.hand].reshape(3, 3)
        self.d.qpos[self.ball_qpos:self.ball_qpos + 3] = (
            self.d.xpos[self.hand] + rotation @ BALL_OFFSET)
        self.d.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [1, 0, 0, 0]
        self.d.eq_active[self.grip] = 1
        mujoco.mj_forward(self.m, self.d)

        self.walk.reset()
        self.released = False
        self.landed = None
        self.launch_velocity = np.zeros(3)
        self.hand_speed = 0.0
        self.scheduled_release_time = None
        self.step_index = 0
        self.start_xy = self.d.qpos[:2].copy()
        self.start_yaw = _yaw_from_quat(self.d.qpos[3:7])
        self.walk_command = self.command.copy()
        self.forward_at_throw_start = None
        self.forward_at_throw_end = None
        self.throw_velocity_samples = 0
        self.throw_positive_velocity_samples = 0
        self.min_throw_forward_velocity = np.inf
        self.max_abs_lateral_error = 0.0
        self.previous_residual = np.zeros(N_ACTIONS, dtype=np.float64)
        self.throw_potential = self._throw_potential()
        return self._observation(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (N_ACTIONS,):
            raise ValueError(
                f"action shape is {action.shape}, expected {(N_ACTIONS,)}")

        # Stage gates which residual channels the policy controls (same 24-D
        # policy throughout, so a checkpoint can be resumed at a later stage):
        #   release: only release timing (23) - pure aiming on the stable nominal
        #            throw, no residual can destabilise the robot.
        #   throw:   right-arm residuals (19:23) + release timing.
        #   full:    all channels including gait/balance residuals.
        control_action = action.copy()
        if self.stage == "release":
            control_action[:23] = 0.0
        elif self.stage == "throw":
            control_action[:19] = 0.0

        # Latch the release time only once the ball is close to launch, so the
        # release action is read from a fresh observation a dozen control steps
        # before release rather than on step 0 (~150 steps early). Committing it
        # on step 0 placed the landing reward ~150 steps downstream of the action
        # that caused it - a credit-assignment gap that made release timing (the
        # main distance lever) destabilise instead of learn. The margin keeps the
        # full +/- release_window reachable (release is never scheduled in the past).
        release_decision_time = (
            self.nominal_release_time - self.release_window - 2.0 * self.dt)
        if (
            self.scheduled_release_time is None
            and self.d.time >= release_decision_time
        ):
            self.scheduled_release_time = (
                self.nominal_release_time
                + self.release_window * control_action[23])

        self._capture_phase_metrics()
        self.walk_command = self._corrected_walk_command()
        walk_target = self.walk.act(self.d, self.walk_command)
        target = self._upper_body_baseline(self.d.time)
        target[self.leg_actuator_ids] = (
            walk_target + LEG_RESIDUAL_SCALE * control_action[0:12])
        target[self.waist_actuator_ids] += (
            WAIST_RESIDUAL_SCALE * control_action[12:15])
        target[self.left_arm_actuator_ids] += (
            LEFT_ARM_RESIDUAL_SCALE * control_action[15:19])
        target[self.right_arm_actuator_ids] += (
            RIGHT_ARM_NOMINAL_OFFSET
            + RIGHT_ARM_RESIDUAL_SCALE * control_action[19:23])
        self.d.ctrl[:] = np.clip(
            target,
            self.m.actuator_ctrlrange[:, 0],
            self.m.actuator_ctrlrange[:, 1],
        )

        release_substep = None
        if not self.released and self.scheduled_release_time is not None:
            remaining = self.scheduled_release_time - self.d.time
            if remaining < self.dt:
                release_substep = int(np.clip(
                    np.ceil(max(remaining - 1e-9, 0.0)
                            / self.m.opt.timestep),
                    0, self.decimation - 1,
                ))

        for substep in range(self.decimation):
            released_now = False
            if release_substep is not None and substep == release_substep:
                self.hand_speed = float(
                    np.linalg.norm(self._hand_velocity()))
                self.d.eq_active[self.grip] = 0
                self.released = True
                mujoco.mj_forward(self.m, self.d)
                released_now = True
            mujoco.mj_step(self.m, self.d)
            if released_now:
                self.launch_velocity = self._ball_vel()
            if (
                self.released
                and self.landed is None
                and self._ball_pos()[2] <= self.landing_z
            ):
                self.landed = self._ball_pos()

        self.step_index += 1
        self._capture_phase_metrics()
        fallen, gravity = self._fall_state()
        terminated = bool(fallen and self.d.time > 0.25)
        truncated = self.step_index >= self.max_steps

        # Time terms use dt so their total is independent of control frequency.
        # The landing term remains a telescoping reduction in target error.
        new_potential = self._throw_potential()
        throw_progress = new_potential - self.throw_potential
        self.throw_potential = new_potential

        (
            distance_walked,
            lateral_drift,
            forward_velocity,
            lateral_velocity,
            heading_error,
        ) = self._line_state()
        velocity_error = (
            (forward_velocity - self.command[0]) ** 2
            + (lateral_velocity - self.command[1]) ** 2
            + 0.25 * (self.d.qvel[5] - self.command[2]) ** 2
        )
        velocity_tracking = float(np.exp(-4.0 * velocity_error))
        upright = float(np.exp(-6.0 * np.dot(gravity[:2], gravity[:2])))
        height_error = float((self.d.qpos[2] - 0.78) ** 2)
        action_rate = float(np.mean(
            np.square(action - self.previous_residual)))
        action_energy = float(np.mean(np.square(action)))
        self.previous_residual = action.copy()

        # End-of-episode accuracy bonus (only if the ball was actually thrown):
        # -new_potential is the current landing miss (landed value once it hits).
        final_miss = -new_potential
        landing_bonus = (
            LANDING_BONUS
            * float(np.exp(-(final_miss / LANDING_BONUS_SCALE) ** 2))
            if (terminated or truncated) and self.released
            else 0.0
        )
        reward_terms = {
            # Dense aim shaping, strengthened; the terminal landing_bonus is the
            # primary driver that makes accurate aiming beat throwing short.
            "throw": 3.0 * throw_progress,
            "landing_bonus": landing_bonus,
            "velocity": velocity_tracking * self.dt,
            "upright": 0.6 * upright * self.dt,
            "height": -3.0 * height_error * self.dt,
            "straight_line": -2.0 * lateral_drift ** 2 * self.dt,
            "heading": -0.6 * heading_error ** 2 * self.dt,
            "vertical_speed": -0.08 * self.d.qvel[2] ** 2 * self.dt,
            # Residuals are deliberately regularised around zero so early
            # TD3 updates do not erase the pretrained gait.
            "action_rate": -0.02 * action_rate,
            "action_energy": -0.01 * action_energy,
            "alive": 0.15 * self.dt,
            "fall": -20.0 if terminated else 0.0,
        }
        reward = float(sum(reward_terms.values()))

        before_throw = (
            self.forward_at_throw_start
            if self.forward_at_throw_start is not None
            else distance_walked
        )
        throw_end = (
            self.forward_at_throw_end
            if self.forward_at_throw_end is not None
            else distance_walked
        )
        during_throw = throw_end - before_throw
        after_throw = distance_walked - throw_end
        positive_fraction = (
            self.throw_positive_velocity_samples
            / max(self.throw_velocity_samples, 1)
        )
        min_throw_velocity = (
            self.min_throw_forward_velocity
            if np.isfinite(self.min_throw_forward_velocity)
            else forward_velocity
        )
        objective_miss = (
            float(-self.throw_potential)
            if terminated or truncated else None
        )
        landing_miss = (
            float(np.linalg.norm(self.landed[:2] - self.target[:2]))
            if self.landed is not None else None
        )
        info = {
            "released": self.released,
            "landed": None if self.landed is None else self.landed.copy(),
            "target": self.target[:2].copy(),
            "objective_miss": objective_miss,
            "landing_miss": landing_miss,
            "fallen": fallen,
            "distance_walked": distance_walked,
            "lateral_drift": lateral_drift,
            "max_abs_lateral_error": self.max_abs_lateral_error,
            "distance_before_throw": before_throw,
            "distance_during_throw": during_throw,
            "distance_after_throw": after_throw,
            "positive_velocity_fraction_during_throw": positive_fraction,
            "min_forward_velocity_during_throw": min_throw_velocity,
            "heading_error": heading_error,
            "walk_command": self.walk_command.copy(),
            "hand_speed": self.hand_speed,
            "launch_velocity": self.launch_velocity.copy(),
            "scheduled_release_time": self.scheduled_release_time,
            "reward_terms": reward_terms,
        }

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )

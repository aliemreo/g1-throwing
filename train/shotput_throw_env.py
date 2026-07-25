"""Learned shot-put task for a walking Unitree G1.

This environment intentionally contains no nominal throwing trajectory and no
calibrated arm offsets. The pretrained recurrent policy remains responsible for
ordinary walking, while TD3 directly commands the whole-body residuals, the
four right-arm joint targets, and the release time.

The only shot-put prior is expressed as reward shaping: before the drive phase,
the hand is rewarded for reaching a neck-side ready region with a raised elbow.
During the drive, forward hand speed, extension, aim, balance, and landing
accuracy are rewarded. The reference pose is never written to ``data.ctrl``.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from walking_throw_env import (
    G,
    LEFT_ARM_RESIDUAL_SCALE,
    LEG_RESIDUAL_SCALE,
    N_ACTIONS,
    RIGHT_ARM_JOINTS,
    WAIST_RESIDUAL_SCALE,
    WalkingThrowEnv,
)


SHOTPUT_ACTION_GROUPS = {
    "leg_balance_residual": (0, 12),
    "waist_residual": (12, 15),
    "left_arm_balance": (15, 19),
    "right_arm_absolute_target": (19, 23),
    "release_time": (23, 24),
}
SHOTPUT_N_ACTIONS = N_ACTIONS

# The zero action corresponds to the model's neutral arm pose. These scales
# cover both a high neck-side preparation and a forward extended release.
RIGHT_ARM_TARGET_SCALE = np.asarray(
    [2.20, 1.00, 1.20, 1.30],
    dtype=np.float64,
)
RIGHT_ARM_RATE_LIMIT = np.asarray(
    [6.0, 5.0, 6.0, 7.0],
    dtype=np.float64,
)

# Hand position relative to the torso frame for a recognisable shot-put ready
# shape: hand behind and to the right of the head, with the elbow raised.
# This is a reward target only, not a control target.
SHOTPUT_READY_HAND_LOCAL = np.asarray(
    [-0.31, -0.25, 0.40],
    dtype=np.float64,
)
SHOTPUT_READY_ELBOW_LOCAL = np.asarray(
    [-0.17, -0.21, 0.29],
    dtype=np.float64,
)


def _ballistic_landing(position, velocity):
    """Return the no-drag XY landing estimate for a point mass."""
    height = max(float(position[2]), 1e-6)
    vertical = float(velocity[2])
    flight = (
        vertical
        + math.sqrt(vertical * vertical + 2.0 * G * height)
    ) / G
    return position[:2] + velocity[:2] * flight


class ShotPutWalkingEnv(WalkingThrowEnv):
    """Walk forward and learn a human-like shot-put action with TD3."""

    def __init__(
        self,
        prep_duration=0.70,
        release_delay=(0.12, 0.55),
        follow_through=0.40,
        stage="arm",
        **kwargs,
    ):
        if stage not in {"arm", "upper", "full"}:
            raise ValueError("stage must be 'arm', 'upper', or 'full'")
        if len(release_delay) != 2:
            raise ValueError("release_delay must contain MIN and MAX")
        release_min, release_max = map(float, release_delay)
        if not 0.05 <= release_min < release_max:
            raise ValueError("release_delay must satisfy 0.05 <= MIN < MAX")

        self._shotput_ready = False
        self.shotput_stage = stage
        self.prep_duration = float(prep_duration)
        self.release_delay = (release_min, release_max)
        self.follow_through = float(follow_through)

        # The parent supplies model loading, walking, target sampling, named
        # addresses, coordinate utilities, and fall/line metrics. Its nominal
        # throw function is never called by this class.
        super().__init__(stage="throw", **kwargs)

        self.torso = self._body_id("torso_link")
        self.right_elbow = self._body_id("right_elbow_link")
        self.right_shoulder = self._body_id(
            "right_shoulder_pitch_link")
        elbow_joint = self._joint_id("right_elbow_joint")
        self.right_elbow_qpos = int(self.m.jnt_qposadr[elbow_joint])
        self.right_arm_neutral = self.q_nominal[
            self.right_arm_actuator_ids].copy()
        self.motion_start = self.throw_start - self.prep_duration
        if self.motion_start < 0.05:
            raise ValueError(
                "throw_start must be greater than prep_duration + 0.05")

        self.n_act = SHOTPUT_N_ACTIONS
        self._shotput_ready = True
        observation, _ = self.reset()
        self.n_obs = int(observation.size)

    # --------------------------------------------------------- observations

    def _torso_local(self, world_position):
        rotation = self.d.xmat[self.torso].reshape(3, 3)
        return rotation.T @ (
            np.asarray(world_position) - self.d.xpos[self.torso])

    def _shotput_state(self):
        hand_position = self.d.xpos[self.hand].copy()
        elbow_position = self.d.xpos[self.right_elbow].copy()
        shoulder_position = self.d.xpos[self.right_shoulder].copy()
        hand_local = self._torso_local(hand_position)
        elbow_local = self._torso_local(elbow_position)
        hand_velocity = self._hand_velocity()

        direction_xy = self.target[:2] - hand_position[:2]
        direction_norm = float(np.linalg.norm(direction_xy))
        if direction_norm < 1e-8:
            direction_xy = np.asarray([1.0, 0.0])
        else:
            direction_xy = direction_xy / direction_norm

        forward_speed = float(np.dot(
            hand_velocity[:2], direction_xy))
        hand_forward = float(np.dot(
            hand_position[:2] - self.d.xpos[self.torso, :2],
            direction_xy,
        ))
        ready_hand_error = float(np.linalg.norm(
            hand_local - SHOTPUT_READY_HAND_LOCAL))
        ready_elbow_error = float(np.linalg.norm(
            elbow_local - SHOTPUT_READY_ELBOW_LOCAL))
        ready_quality = float(np.exp(
            -10.0 * ready_hand_error * ready_hand_error
            -4.0 * ready_elbow_error * ready_elbow_error
        ))
        extension = float(np.linalg.norm(
            hand_position - shoulder_position))

        return {
            "hand_position": hand_position,
            "elbow_position": elbow_position,
            "hand_local": hand_local,
            "elbow_local": elbow_local,
            "hand_velocity": hand_velocity,
            "direction_xy": direction_xy,
            "forward_speed": forward_speed,
            "upward_speed": float(hand_velocity[2]),
            "hand_forward": hand_forward,
            "ready_hand_error": ready_hand_error,
            "ready_elbow_error": ready_elbow_error,
            "ready_quality": ready_quality,
            "extension": extension,
            "elbow_angle": float(self.d.qpos[self.right_elbow_qpos]),
        }

    def _observation(self):
        base = super()._observation()
        if not self._shotput_ready:
            return base

        state = self._shotput_state()
        phase_span = (
            self.prep_duration + self.release_delay[1]
        )
        motion_phase = np.clip(
            (self.d.time - self.motion_start) / phase_span,
            -1.0,
            2.0,
        )
        release_remaining = (
            0.0
            if self.scheduled_release_time is None
            else np.clip(
                (self.scheduled_release_time - self.d.time)
                / max(self.release_delay[1], 1e-6),
                -1.0,
                1.0,
            )
        )
        extra = np.concatenate([
            state["hand_local"],
            state["elbow_local"],
            state["hand_velocity"] * 0.20,
            SHOTPUT_READY_HAND_LOCAL - state["hand_local"],
            [
                motion_phase,
                release_remaining,
                float(getattr(
                    self, "best_ready_quality", 0.0)),
            ],
        ])
        return np.concatenate([base, extra]).astype(np.float32)

    # --------------------------------------------------------------- reset

    def reset(self, seed=None):
        observation, info = super().reset(seed=seed)
        if not self._shotput_ready:
            return observation, info

        self.right_arm_target = self.right_arm_neutral.copy()
        self.previous_residual = np.zeros(
            SHOTPUT_N_ACTIONS, dtype=np.float64)
        self.previous_ready_error = np.inf
        self.previous_aim_error = np.inf
        self.previous_hand_forward = 0.0
        self.previous_ballistic_miss = None
        self.best_ready_quality = 0.0
        self.best_ready_hand_error = np.inf
        self.release_ready_quality = 0.0
        self.release_forward_speed = 0.0
        self.release_upward_speed = 0.0
        self.release_hand_height = 0.0
        self.release_elbow_angle = float(
            self.d.qpos[self.right_elbow_qpos])
        self.release_extension = 0.0
        self.release_style_score = 0.0
        self.release_style_valid = False
        self.release_step = None
        self.scheduled_release_time = None
        self.throw_potential = None
        self.forward_at_throw_start = None
        self.forward_at_throw_end = None
        return self._observation(), info

    # -------------------------------------------------------------- metrics

    def _capture_phase_metrics(self):
        if not self._shotput_ready:
            return super()._capture_phase_metrics()

        time_s = float(self.d.time)
        (
            forward_distance,
            lateral_error,
            forward_velocity,
            _,
            _,
        ) = self._line_state()
        self.max_abs_lateral_error = max(
            self.max_abs_lateral_error, abs(lateral_error))

        if (
            self.forward_at_throw_start is None
            and time_s >= self.motion_start - 1e-9
        ):
            self.forward_at_throw_start = forward_distance

        motion_end = (
            self.scheduled_release_time
            if self.scheduled_release_time is not None
            else self.throw_start + self.release_delay[1]
        ) + self.follow_through
        if (
            self.forward_at_throw_end is None
            and time_s >= motion_end - 1e-9
        ):
            self.forward_at_throw_end = forward_distance

        if self.motion_start <= time_s <= motion_end:
            self.throw_velocity_samples += 1
            self.throw_positive_velocity_samples += int(
                forward_velocity > 0.02)
            self.min_throw_forward_velocity = min(
                self.min_throw_forward_velocity,
                forward_velocity,
            )

    def _active_action_mask(self):
        mask = np.zeros(SHOTPUT_N_ACTIONS, dtype=bool)
        if self.shotput_stage == "arm":
            mask[19:24] = True
        elif self.shotput_stage == "upper":
            mask[12:24] = True
        else:
            mask[:] = True
        return mask

    def _style_score(self):
        ready = float(np.clip(
            self.release_ready_quality, 0.0, 1.0))
        speed = float(np.clip(
            self.release_forward_speed / 3.5, 0.0, 1.0))
        height = float(np.clip(
            (self.release_hand_height - 0.82) / 0.35,
            0.0,
            1.0,
        ))
        extension = float(np.clip(
            (1.10 - abs(self.release_elbow_angle)) / 1.10,
            0.0,
            1.0,
        ))
        score = (
            0.30 * ready
            + 0.35 * speed
            + 0.20 * height
            + 0.15 * extension
        )
        valid = bool(
            self.released
            and self.best_ready_hand_error < 0.32
            and self.release_forward_speed > 1.50
            and self.release_hand_height > 1.00
            and abs(self.release_elbow_angle) < 1.00
        )
        return float(score), valid

    def _aim_error_from_hand(self, state):
        predicted = _ballistic_landing(
            state["hand_position"],
            state["hand_velocity"],
        )
        return float(np.linalg.norm(
            predicted - self.target[:2]))

    # --------------------------------------------------------------- step

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (SHOTPUT_N_ACTIONS,):
            raise ValueError(
                f"action shape is {action.shape}, "
                f"expected {(SHOTPUT_N_ACTIONS,)}")

        active_mask = self._active_action_mask()
        control_action = action.copy()
        control_action[~active_mask] = 0.0
        phase_active = self.d.time >= self.motion_start

        if (
            phase_active
            and self.scheduled_release_time is None
        ):
            release_fraction = 0.5 * (
                control_action[23] + 1.0)
            release_offset = np.interp(
                release_fraction,
                [0.0, 1.0],
                self.release_delay,
            )
            self.scheduled_release_time = (
                self.throw_start + release_offset)

        self._capture_phase_metrics()
        self.walk_command = self._corrected_walk_command()
        walk_target = self.walk.act(self.d, self.walk_command)
        target = self.q_nominal.copy()
        target[self.leg_actuator_ids] = walk_target

        if phase_active:
            if self.shotput_stage == "full":
                target[self.leg_actuator_ids] += (
                    LEG_RESIDUAL_SCALE * control_action[0:12])

            if self.shotput_stage in {"upper", "full"}:
                target[self.waist_actuator_ids] += (
                    WAIST_RESIDUAL_SCALE * control_action[12:15])
                target[self.left_arm_actuator_ids] += (
                    LEFT_ARM_RESIDUAL_SCALE * control_action[15:19])

            desired_right_arm = (
                self.right_arm_neutral
                + RIGHT_ARM_TARGET_SCALE * control_action[19:23]
            )
            max_change = RIGHT_ARM_RATE_LIMIT * self.dt
            self.right_arm_target += np.clip(
                desired_right_arm - self.right_arm_target,
                -max_change,
                max_change,
            )
        else:
            self.right_arm_target = self.right_arm_neutral.copy()

        target[self.right_arm_actuator_ids] = self.right_arm_target
        self.d.ctrl[:] = np.clip(
            target,
            self.m.actuator_ctrlrange[:, 0],
            self.m.actuator_ctrlrange[:, 1],
        )

        release_substep = None
        if (
            not self.released
            and self.scheduled_release_time is not None
        ):
            remaining = self.scheduled_release_time - self.d.time
            if remaining < self.dt:
                release_substep = int(np.clip(
                    np.ceil(
                        max(remaining - 1e-9, 0.0)
                        / self.m.opt.timestep
                    ),
                    0,
                    self.decimation - 1,
                ))

        released_this_step = False
        landed_this_step = False
        for substep in range(self.decimation):
            released_now = False
            if (
                release_substep is not None
                and substep == release_substep
            ):
                release_state = self._shotput_state()
                self.hand_speed = float(np.linalg.norm(
                    release_state["hand_velocity"]))
                self.release_forward_speed = float(
                    release_state["forward_speed"])
                self.release_upward_speed = float(
                    release_state["upward_speed"])
                self.release_hand_height = float(
                    release_state["hand_position"][2])
                self.release_elbow_angle = float(
                    release_state["elbow_angle"])
                self.release_extension = float(
                    release_state["extension"])
                self.release_ready_quality = float(
                    self.best_ready_quality)
                self.d.eq_active[self.grip] = 0
                self.released = True
                self.release_step = self.step_index
                mujoco.mj_forward(self.m, self.d)
                released_now = True
                released_this_step = True

            mujoco.mj_step(self.m, self.d)
            if released_now:
                self.launch_velocity = self._ball_vel()
            if (
                self.released
                and self.landed is None
                and self._ball_pos()[2] <= self.ball_radius + 0.003
            ):
                self.landed = self._ball_pos()
                landed_this_step = True

        self.step_index += 1
        self._capture_phase_metrics()
        fallen, gravity = self._fall_state()
        terminated = bool(fallen and self.d.time > 0.25)
        truncated = self.step_index >= self.max_steps

        state = self._shotput_state()
        if phase_active and not self.released:
            self.best_ready_quality = max(
                self.best_ready_quality,
                state["ready_quality"],
            )
            self.best_ready_hand_error = min(
                self.best_ready_hand_error,
                state["ready_hand_error"],
            )

        ready_progress = (
            0.0
            if not np.isfinite(self.previous_ready_error)
            else self.previous_ready_error
            - state["ready_hand_error"]
        )
        self.previous_ready_error = state["ready_hand_error"]

        aim_error = self._aim_error_from_hand(state)
        aim_progress = (
            0.0
            if not np.isfinite(self.previous_aim_error)
            else self.previous_aim_error - aim_error
        )
        self.previous_aim_error = aim_error
        hand_forward_progress = (
            state["hand_forward"] - self.previous_hand_forward)
        self.previous_hand_forward = state["hand_forward"]

        ballistic_progress = 0.0
        if self.released:
            ballistic_miss = -self._throw_potential()
            if self.previous_ballistic_miss is not None:
                ballistic_progress = (
                    self.previous_ballistic_miss
                    - ballistic_miss
                )
            self.previous_ballistic_miss = ballistic_miss

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
            + 0.25 * (
                self.d.qvel[5] - self.command[2]
            ) ** 2
        )
        velocity_tracking = float(np.exp(
            -4.0 * velocity_error))
        upright = float(np.exp(
            -6.0 * np.dot(gravity[:2], gravity[:2])))
        height_error = float((self.d.qpos[2] - 0.78) ** 2)

        action_delta = action - self.previous_residual
        action_rate = float(np.mean(np.square(
            action_delta[active_mask])))
        action_energy = float(np.mean(np.square(
            action[active_mask])))
        self.previous_residual = action.copy()

        in_preparation = bool(
            self.motion_start <= self.d.time < self.throw_start
            and not self.released
        )
        in_drive = bool(
            self.d.time >= self.throw_start
            and not self.released
        )
        preparation_reward = (
            1.8 * state["ready_quality"] * self.dt
            + 2.0 * np.clip(ready_progress, -0.06, 0.06)
            if in_preparation else 0.0
        )
        elbow_reward = (
            0.45
            * np.exp(
                -12.0
                * (
                    state["elbow_local"][2]
                    - SHOTPUT_READY_ELBOW_LOCAL[2]
                ) ** 2
            )
            * self.dt
            if in_preparation else 0.0
        )
        drive_reward = (
            0.55
            * np.clip(state["forward_speed"], 0.0, 6.0)
            * self.dt
            + 0.65 * np.clip(aim_progress, -0.20, 0.20)
            + 0.30
            * np.clip(hand_forward_progress, -0.10, 0.10)
            if in_drive else 0.0
        )

        release_reward = 0.0
        if released_this_step:
            self.release_style_score, self.release_style_valid = (
                self._style_score())
            release_reward = (
                1.2 * np.clip(
                    self.release_forward_speed, -2.0, 6.0)
                + 2.5 * self.release_style_score
            )

        landing_miss = (
            float(np.linalg.norm(
                self.landed[:2] - self.target[:2]))
            if self.landed is not None else None
        )
        accuracy_reward = 0.0
        if landed_this_step and landing_miss is not None:
            accuracy_reward = (
                15.0 * np.exp(
                    -1.35 * landing_miss * landing_miss)
                - 1.5 * landing_miss
            )

        no_release_penalty = (
            -12.0
            if (terminated or truncated) and not self.released
            else 0.0
        )
        no_landing_penalty = (
            -5.0
            if (
                terminated or truncated
                and self.released
                and self.landed is None
            )
            else 0.0
        )
        reward_terms = {
            "shotput_preparation": float(preparation_reward),
            "raised_elbow": float(elbow_reward),
            "forward_drive": float(drive_reward),
            "release_quality": float(release_reward),
            "ballistic_progress": float(
                2.5 * np.clip(
                    ballistic_progress, -0.5, 0.5)),
            "landing_accuracy": float(accuracy_reward),
            "velocity": velocity_tracking * self.dt,
            "upright": 0.6 * upright * self.dt,
            "height": -3.0 * height_error * self.dt,
            "straight_line": (
                -2.0 * lateral_drift * lateral_drift * self.dt
            ),
            "heading": (
                -0.6 * heading_error * heading_error * self.dt
            ),
            "vertical_speed": (
                -0.08 * self.d.qvel[2] ** 2 * self.dt
            ),
            "action_rate": -0.012 * action_rate,
            "action_energy": -0.004 * action_energy,
            "alive": 0.15 * self.dt,
            "no_release": no_release_penalty,
            "no_landing": no_landing_penalty,
            "fall": -25.0 if terminated else 0.0,
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

        if terminated or truncated:
            if landing_miss is not None:
                objective_miss = landing_miss
            elif self.released:
                objective_miss = -self._throw_potential()
            else:
                objective_miss = float(np.linalg.norm(
                    self.target[:2] - self.d.qpos[:2]))
        else:
            objective_miss = None

        info = {
            "controller_source": (
                "td3 actions; no nominal throw trajectory"),
            "released": self.released,
            "landed": (
                None if self.landed is None
                else self.landed.copy()
            ),
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
            "shotput_ready_error": (
                self.best_ready_hand_error
                if np.isfinite(self.best_ready_hand_error)
                else state["ready_hand_error"]
            ),
            "shotput_ready_quality": self.best_ready_quality,
            "release_forward_speed": self.release_forward_speed,
            "release_upward_speed": self.release_upward_speed,
            "release_hand_height": self.release_hand_height,
            "release_elbow_angle": self.release_elbow_angle,
            "release_extension": self.release_extension,
            "shotput_style_score": self.release_style_score,
            "shotput_style_valid": self.release_style_valid,
            "aim_error": aim_error,
            "stage": self.shotput_stage,
            "reward_terms": reward_terms,
        }
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )

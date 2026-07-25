"""Adapter for the pretrained 12-DoF recurrent G1 walking controller.

The policy comes from HusseinLezzaik/unitree-g1-bipedal-rl-walk. Its exact
47-value observation convention and 50 Hz control period are preserved here.
The policy produces leg joint position targets; MuJoCo applies the same PD
gains used by the original sim-to-sim deployment.
"""

from pathlib import Path

import mujoco
import numpy as np
import torch


DEFAULT_POLICY = (
    Path(__file__).resolve().parent.parent
    / "pretrained" / "g1_walk_policy_lstm.pt"
)

LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

DEFAULT_ANGLES = np.asarray(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float64,
)
KPS = np.asarray(
    [100, 100, 100, 150, 40, 40,
     100, 100, 100, 150, 40, 40],
    dtype=np.float64,
)
KDS = np.asarray(
    [2, 2, 2, 4, 2, 2,
     2, 2, 2, 4, 2, 2],
    dtype=np.float64,
)

ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
COMMAND_SCALE = np.asarray([2.0, 2.0, 0.25], dtype=np.float32)
GAIT_PERIOD = 0.8
OBS_SIZE = 47


def projected_gravity(quat):
    """Return gravity direction in the body frame for MuJoCo [w,x,y,z]."""
    qw, qx, qy, qz = quat
    return np.asarray([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float64)


def joint_addresses(model, names):
    """Resolve named hinge joints to qpos and qvel scalar addresses."""
    qpos, qvel = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"walking joint is missing from model: {name}")
        qpos.append(model.jnt_qposadr[jid])
        qvel.append(model.jnt_dofadr[jid])
    return np.asarray(qpos, dtype=np.int32), np.asarray(qvel, dtype=np.int32)


class WalkPolicy:
    """Stateful TorchScript LSTM walking controller."""

    def __init__(self, model, policy_path=DEFAULT_POLICY, device="cpu"):
        self.path = Path(policy_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"walking policy not found: {self.path}")
        self.device = torch.device(device)
        self.policy = torch.jit.load(
            str(self.path), map_location=self.device).eval()
        if "reset_memory" not in self.policy._c._method_names():
            raise ValueError("walking policy has no reset_memory method")

        self.qpos_ids, self.qvel_ids = joint_addresses(model, LEG_JOINTS)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.last_target = DEFAULT_ANGLES.copy()
        self.reset()

    def reset(self):
        self.policy.reset_memory()
        self.previous_action.fill(0.0)
        self.last_target = DEFAULT_ANGLES.copy()

    def observation(self, data, command):
        phase = (data.time % GAIT_PERIOD) / GAIT_PERIOD
        qj = ((data.qpos[self.qpos_ids] - DEFAULT_ANGLES)
              * DOF_POS_SCALE)
        dqj = data.qvel[self.qvel_ids] * DOF_VEL_SCALE

        obs = np.concatenate([
            data.qvel[3:6] * ANG_VEL_SCALE,
            projected_gravity(data.qpos[3:7]),
            np.asarray(command, dtype=np.float64) * COMMAND_SCALE,
            qj,
            dqj,
            self.previous_action,
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
        ]).astype(np.float32)
        if obs.shape != (OBS_SIZE,):
            raise RuntimeError(f"walk observation is {obs.shape}, expected 47")
        return obs

    def act(self, data, command):
        obs = self.observation(data, command)
        tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            action = self.policy(tensor).squeeze(0).cpu().numpy()
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError("walking policy returned an invalid action")
        self.previous_action = action.astype(np.float32, copy=True)
        self.last_target = DEFAULT_ANGLES + ACTION_SCALE * action
        return self.last_target.copy()


def configure_leg_pd(model):
    """Make the Menagerie position actuators match the walking PD controller."""
    actuator_ids = []
    for index, name in enumerate(LEG_JOINTS):
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"walking actuator is missing from model: {name}")
        actuator_ids.append(aid)
        model.actuator_gainprm[aid, 0] = KPS[index]
        model.actuator_biasprm[aid, 1] = -KPS[index]
        model.actuator_biasprm[aid, 2] = -KDS[index]
    return np.asarray(actuator_ids, dtype=np.int32)

import mujoco
import numpy as np
from pathlib import Path

path = Path(__file__).parent.parent / "models" / "throw_scene.xml"
assert path.exists(), f"not found: {path}"

m = mujoco.MjModel.from_xml_path(str(path))
d = mujoco.MjData(m)
print(f"nq={m.nq} nv={m.nv} nu={m.nu}")   # expect 43 / 41 / 29

OFFSET = np.array([0.145, 0.0, 0.0])   # must match relpose in the XML

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
ball_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]

R = d.xmat[hand].reshape(3, 3)
d.qpos[ball_adr:ball_adr+3] = d.xpos[hand] + R @ OFFSET
d.qpos[ball_adr+3:ball_adr+7] = [1, 0, 0, 0]

d.ctrl[:] = d.qpos[7:7+m.nu]   # hold the standing pose

for i in range(2000):
    mujoco.mj_step(m, d)
    if i % 400 == 0:
        ball = d.qpos[ball_adr:ball_adr+3]
        print(f"t={d.time:5.2f}  ball={ball}  pelvis_z={d.qpos[2]:.3f}")
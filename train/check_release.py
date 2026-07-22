import mujoco
import numpy as np
from pathlib import Path

path = Path(__file__).parent.parent / "models" / "throw_scene.xml"
m = mujoco.MjModel.from_xml_path(str(path))
d = mujoco.MjData(m)

OFFSET = np.array([0.145, 0.0, 0.0])

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
ball_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
R = d.xmat[hand].reshape(3, 3)
d.qpos[ball_adr:ball_adr+3] = d.xpos[hand] + R @ OFFSET
d.qpos[ball_adr+3:ball_adr+7] = [1, 0, 0, 0]
d.ctrl[:] = d.qpos[7:7+m.nu]

released = False
while d.time < 3.0:
    if not released and d.time >= 1.0:
        d.eq_active[grip] = 0
        released = True
        print(f">>> released at t={d.time:.2f}, ball_z={d.qpos[ball_adr+2]:.3f}")

    mujoco.mj_step(m, d)

    if abs(d.time % 0.25) < 1e-9 or d.time < 1e-9:
        print(f"t={d.time:5.2f}  ball_z={d.qpos[ball_adr+2]:.3f}  "
              f"held={'Y' if d.eq_active[grip] else 'N'}")

print(f"\nfinal ball position: {d.qpos[ball_adr:ball_adr+3]}")
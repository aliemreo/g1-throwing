import mujoco
from pathlib import Path

path = Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
m = mujoco.MjModel.from_xml_path(str(path))
d = mujoco.MjData(m)

mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)          # compute body positions without stepping

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
print("hand world pos:", d.xpos[hand])
print("pelvis world pos:", d.xpos[1])
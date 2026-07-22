import mujoco
from pathlib import Path

path = Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
m = mujoco.MjModel.from_xml_path(str(path))

hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")

print("geoms on right_wrist_yaw_link (positions are in the wrist frame):")
for g in range(m.ngeom):
    if m.geom_bodyid[g] == hand:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        print(f"  {name}  local_pos={m.geom_pos[g]}  size={m.geom_size[g]}")

print(f"\nbody ipos (center of mass, wrist frame): {m.body_ipos[hand]}")
import mujoco
from pathlib import Path

path = Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1.xml"
m = mujoco.MjModel.from_xml_path(str(path))

print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  nkey={m.nkey}\n")

print("--- BODIES ---")
for i in range(m.nbody):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i))

print("\n--- ACTUATORS ---")
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i:2d} {name:32s} ctrl={m.actuator_ctrlrange[i]} force={m.actuator_forcerange[i]}")

print("\n--- KEYFRAMES ---")
for i in range(m.nkey):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i))
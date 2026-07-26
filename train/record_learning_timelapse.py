"""Learning-process timelapse for the from-scratch (no-baseline) TD3 throw.

Renders the SAME throw at successive training checkpoints into one MP4, so you
watch RL discover the throw: it starts unable to throw (ball drops at the feet)
and progressively learns to swing and land on the target. Styled to match the
other out_videos (elevated side view, flat green target disk, black HUD bar).

    python record_learning_timelapse.py --run td3_throw_human_fine
    python record_learning_timelapse.py --run td3_throw_human --ckpts 100000 200000

Needs a video writer:  pip install imageio imageio-ffmpeg  (Pillow for labels)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_env import make_env
from train_td3 import THROW_JOINTS, RELEASE_TIME_RANGE

try:
    import imageio.v2 as imageio
except ImportError:
    raise SystemExit("pip install imageio imageio-ffmpeg")

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def label_frame(frame, line1, line2=None):
    """Black HUD bar: white line1, light-blue line2 (matches the other videos)."""
    if not _HAVE_PIL:
        return frame
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    bar = max(34, int(img.height * (0.12 if line2 else 0.07)))
    draw.rectangle([0, 0, img.width, bar], fill=(0, 0, 0))
    draw.text((int(img.width * 0.012), int(bar * 0.08)), line1,
              fill=(255, 255, 255), font=_font(int(bar * (0.42 if line2 else 0.55))))
    if line2:
        draw.text((int(img.width * 0.012), int(bar * 0.55)), line2,
                  fill=(180, 220, 255), font=_font(int(bar * 0.36)))
    return np.asarray(img)


def style_target(m):
    """Keep the flat green ground disk but make it bright and a touch larger, so
    it reads clearly under the elevated side camera (visual only)."""
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
    if tid < 0:
        return
    m.site_size[tid] = [0.35, 0.006, 0.0]
    m.site_rgba[tid] = [0.05, 0.9, 0.12, 0.9]


def setup_side_camera(m):
    """Reposition the scene 'side' camera (same tilt as the walk_throw videos)
    further back, so the robot at 0 and the target at 4 m are both framed with
    the ground - and the flat target disk - visible."""
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "side")
    if cid >= 0:
        m.cam_pos[cid] = [2.1, -6.8, 1.75]
    return cid


def step_label(n):
    return "Before training" if n is None else f"Training: {n:,} steps"


def rollout(writer, run_dir, ckpt, vecnorm_name, env_kwargs, args, line1):
    """Render one deterministic throw, writing labelled frames; return miss."""
    venv = DummyVecEnv([make_env(0, seed=123, **env_kwargs)])
    if ckpt is not None:
        venv = VecNormalize.load(str(run_dir / vecnorm_name), venv)
        venv.training = False
        venv.norm_reward = False
        model = TD3.load(str(run_dir / ckpt), device="cpu")
    raw = venv.venv.envs[0].env if isinstance(venv, VecNormalize) else venv.envs[0].env
    raw.m.vis.global_.offwidth = args.width
    raw.m.vis.global_.offheight = args.height
    raw.m.vis.quality.offsamples = 8
    raw.m.vis.quality.shadowsize = 8192
    raw.m.vis.headlight.ambient = [0.45, 0.45, 0.45]
    renderer = mujoco.Renderer(raw.m, height=args.height, width=args.width)

    obs = venv.reset()
    style_target(raw.m)
    cam_id = setup_side_camera(raw.m)
    tgt = float(np.linalg.norm(raw.target[:2]))

    frame, k = None, 0
    while True:
        if ckpt is None:
            action = np.zeros((1,) + venv.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        renderer.update_scene(raw.d, camera=cam_id)
        frame = renderer.render()
        if k % max(1, args.stride) == 0:
            writer.append_data(label_frame(frame, line1, f"Target: {tgt:.1f} m"))
        k += 1
        if done[0]:
            miss = float(infos[0]["objective_miss"])
            landed = infos[0].get("landed")
            break
    renderer.close()
    venv.close()

    downrange = float(landed[0]) if landed is not None else 0.0
    if ckpt is None:
        result = f"Target: {tgt:.1f} m   |   ball drops at the feet (no throw)"
    else:
        tag = "HIT" if miss < 0.5 else "miss"
        result = (f"Target: {tgt:.1f} m   |   threw {downrange:.2f} m   "
                  f"({tag} {miss:.2f} m)")
    held = label_frame(frame, line1, result)
    for _ in range(int(args.hold * args.fps)):
        writer.append_data(held)
    print(f"{line1:>26}: threw {downrange:.2f} m  miss {miss:.2f} m")
    return miss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="td3_throw_human_fine")
    p.add_argument("--ckpts", type=int, nargs="*", default=None,
                   help="step counts to show; default: every checkpoint in the run")
    p.add_argument("--baseline", action="store_true", default=True)
    p.add_argument("--no-baseline", dest="baseline", action="store_false")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--hold", type=float, default=1.2)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_kwargs = dict(
        fixed_target=list(config.get("target", [4.0, 0.0])), freeze_legs=True,
        action_joints=THROW_JOINTS, release_time_range=RELEASE_TIME_RANGE,
        action_baseline=None, elbow_min=config.get("elbow_min", None))

    if args.ckpts:
        ckpt_steps = sorted(args.ckpts)
    else:
        ckpt_steps = sorted(int(q.stem.split("_")[1])
                            for q in run_dir.glob("ckpt_*_steps.zip"))
    if not ckpt_steps:
        raise SystemExit(f"no ckpt_*_steps.zip checkpoints in {run_dir}")

    out = Path(args.out) if args.out else run_dir / "plots" / "learning_timelapse.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out), fps=args.fps, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p", output_params=["-crf", "16", "-preset", "slow"])

    if args.baseline:
        rollout(writer, run_dir, None, None, env_kwargs, args, step_label(None))
    for n in ckpt_steps:
        rollout(writer, run_dir, f"ckpt_{n}_steps.zip",
                f"ckpt_vecnormalize_{n}_steps.pkl", env_kwargs, args, step_label(n))
    writer.close()
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()

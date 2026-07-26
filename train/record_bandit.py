"""Render the changing-target aiming controller (bandit) to an MP4.

Offscreen render (no interactive window) of the trained release-timing policy
aiming at a sweep of target distances. Reuses the exact env + action logic from
view_bandit.py so the video matches the live sim.

    python record_bandit.py --run bandit_bigball --out bandit_aiming.mp4
    python record_bandit.py --run bandit_bigball --targets 3.3 4.0 4.6 5.0

Needs a video writer:  pip install imageio imageio-ffmpeg
On-frame labels are optional:  pip install Pillow
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from stable_baselines3 import TD3

from release_bandit_env import RELEASE_INDEX
from walking_throw_env import N_ACTIONS, WalkingThrowEnv

try:
    import imageio.v2 as imageio
except ImportError:
    raise SystemExit(
        "Video writer missing. Install it with:\n"
        "    pip install imageio imageio-ffmpeg")

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


def label_frame(frame, text):
    """Draw a caption bar on top of the frame (no-op without Pillow)."""
    if not _HAVE_PIL:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    bar = max(28, int(img.height * 0.07))
    draw.rectangle([0, 0, img.width, bar], fill=(0, 0, 0))
    draw.text((int(img.width * 0.012), int(bar * 0.18)), text,
              fill=(255, 255, 255), font=_font(int(bar * 0.55)))
    return np.asarray(img)


def render_frame(renderer, data, cam_id):
    if cam_id >= 0:
        renderer.update_scene(data, camera=cam_id)
    else:
        renderer.update_scene(data)
    return renderer.render()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="bandit_bigball")
    p.add_argument("--ckpt", default="best_model")
    p.add_argument("--targets", nargs="*", type=float,
                   default=[3.3, 3.7, 4.1, 4.5, 5.0])
    p.add_argument("--out", default="bandit_aiming.mp4")
    p.add_argument("--cam", default="follow")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=None,
                   help="default: real-time (1/dt)")
    p.add_argument("--hold", type=float, default=1.2,
                   help="seconds to freeze on the landing of each throw")
    args = p.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = TD3.load(str(run_dir / args.ckpt), device="cpu")

    env = WalkingThrowEnv(
        stage="release",
        release_window=config.get("release_window", 0.0055),
        release_center_offset=config.get("release_center_offset", 0.781),
        max_seconds=5.0,
    )
    # Enlarge the offscreen framebuffer to the requested resolution (the default
    # is only 640x480, which would reject a 720p render).
    env.m.vis.global_.offwidth = args.width
    env.m.vis.global_.offheight = args.height

    fps = args.fps or int(round(1.0 / env.dt))
    hold_frames = int(args.hold * fps)
    renderer = mujoco.Renderer(env.m, height=args.height, width=args.width)
    cam_id = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)

    writer = imageio.get_writer(args.out, fps=fps, macro_block_size=None)
    for target in args.targets:
        env.fixed_target = np.asarray([target, 0.0], dtype=np.float64)
        env.reset()
        distance = float(np.linalg.norm(env.target[:2]))
        obs = np.array([[distance - 4.0]], dtype=np.float32)
        release, _ = model.predict(obs, deterministic=True)
        action = np.zeros(N_ACTIONS, dtype=np.float64)
        action[RELEASE_INDEX] = float(release[0][0])

        info, frame = {}, None
        while True:
            _, _, terminated, truncated, info = env.step(action)
            frame = render_frame(renderer, env.d, cam_id)
            writer.append_data(label_frame(frame, f"Target: {distance:.2f} m"))
            if terminated or truncated or env.landed is not None:
                break

        # We stop as soon as the ball lands, before the episode formally
        # terminates, so info["objective_miss"] is still None here. Compute the
        # miss straight from the landing position (as release_bandit_env does).
        if env.landed is not None:
            threw = f"{env.landed[0]:.2f} m"
            miss = float(np.linalg.norm(env.landed[:2] - env.target[:2]))
        else:
            threw = "n/a"
            fallback = info.get("objective_miss")
            miss = float(fallback) if fallback is not None else float("nan")
        tag = "HIT" if miss < 0.5 else "miss"
        caption = (f"Target: {distance:.2f} m   |   threw to {threw}   "
                   f"({tag} by {miss:.2f} m)")
        held = label_frame(frame, caption)
        for _ in range(hold_frames):
            writer.append_data(held)
        print(f"target={distance:.2f} m -> {threw} ({tag} by {miss:.2f} m)")

    writer.close()
    renderer.close()
    print(f"wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()

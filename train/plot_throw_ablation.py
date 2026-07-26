"""Ablation: did RL learn to THROW (from scratch)?  no-RL vs RL.

The pinned throw was trained with train_td3.py --no-warmstart, i.e. with NO
hand-written throw motion - RL had to discover the swing from reward. This shows:
  - No RL (zero action): the arm never swings, the ball drops at the feet.
  - RL: the learned swing throws the ball downrange toward the target.

    python plot_throw_ablation.py --run td3_throw_scratch

Writes runs/<run>/plots/throw_learned_ablation.png (top-down landing map).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_env import make_env
from train_td3 import THROW_JOINTS, RELEASE_TIME_RANGE

BASE_C = "#9ca3af"
RL_C = "#2563eb"
TARGET_C = "#16a34a"


def landing_of(venv, model=None):
    """Run one deterministic episode -> (landing_xy, miss)."""
    obs = venv.reset()
    while True:
        if model is None:
            action = np.zeros(
                (venv.num_envs,) + venv.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = venv.step(action)
        if done[0]:
            landed = info[0].get("landed")
            xy = (np.asarray(landed[:2]) if landed is not None
                  else np.array([np.nan, np.nan]))
            return xy, float(info[0]["objective_miss"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="td3_throw_scratch")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    target = np.asarray(config.get("target", [4.0, 0.0]), dtype=float)

    env_kwargs = dict(
        fixed_target=list(target), freeze_legs=True, action_joints=THROW_JOINTS,
        release_time_range=RELEASE_TIME_RANGE, action_baseline=None,
        # Must match training, or an elbow-clamp policy behaves differently.
        elbow_min=config.get("elbow_min", None))

    # Baseline: raw env, zero action (no normalization needed).
    base_venv = DummyVecEnv([make_env(0, seed=7, **env_kwargs)])
    base_xy, base_miss = landing_of(base_venv, model=None)

    # RL: normalized obs + trained policy.
    rl_venv = DummyVecEnv([make_env(0, seed=7, **env_kwargs)])
    rl_venv = VecNormalize.load(
        str(run_dir / "best_vecnorm.pkl"), rl_venv)
    rl_venv.training = False
    rl_venv.norm_reward = False
    model = TD3.load(str(run_dir / "best_model"), device=args.device)
    rl_xy, rl_miss = landing_of(rl_venv, model=model)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
    fig, ax = plt.subplots(figsize=(9.5, 4.6))

    # Success zone around the target (0.5 m).
    ax.add_patch(plt.Circle(tuple(target), 0.5, color=TARGET_C, alpha=0.15,
                            zorder=1))
    ax.plot(*target, marker="*", ms=20, color=TARGET_C, zorder=5,
            label=f"target ({target[0]:.0f} m)")
    ax.plot(0, 0, marker="s", ms=11, color="#111827", zorder=5,
            label="robot (start)")
    ax.plot(base_xy[0], base_xy[1], "o", ms=13, color=BASE_C, zorder=5,
            label=f"No RL - ball drops  (miss {base_miss:.2f} m)")
    ax.plot(rl_xy[0], rl_xy[1], "o", ms=13, color=RL_C, zorder=5,
            label=f"RL - learned throw  (miss {rl_miss:.2f} m)")
    # release-to-landing arrows
    ax.annotate("", xy=(base_xy[0], base_xy[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=BASE_C, lw=1.5))
    ax.annotate("", xy=(rl_xy[0], rl_xy[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=RL_C, lw=1.5))

    ax.set_xlim(-0.6, max(target[0], rl_xy[0]) + 1.0)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("downrange distance [m]")
    ax.set_ylabel("lateral [m]")
    ax.set_title("Did RL learn to THROW?  (trained from scratch, no scripted swing)")
    ax.legend(loc="upper left", fontsize=9)

    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.5, 0.02,
        "No RL: the arm never swings - the ball drops at the robot's feet "
        f"({base_miss:.1f} m short).\nRL (trained only on a landing-error "
        f"reward): discovered a throwing swing from scratch, sending the ball "
        f"~{rl_xy[0]:.1f} m downrange.",
        ha="center", va="bottom", fontsize=10, color="#374151",
        bbox=dict(boxstyle="round", fc="#f9fafb", ec="#e5e7eb"))

    out = run_dir / "plots" / "throw_learned_ablation.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  no-RL baseline: landed ({base_xy[0]:.2f}, {base_xy[1]:.2f}), "
          f"miss {base_miss:.2f} m")
    print(f"  RL learned    : landed ({rl_xy[0]:.2f}, {rl_xy[1]:.2f}), "
          f"miss {rl_miss:.2f} m")


if __name__ == "__main__":
    main()

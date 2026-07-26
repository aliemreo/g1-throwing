"""Train the release-timing aim as a one-step contextual bandit (TD3).

Each env step is a full walk+throw, so the release action is never diluted and
the policy can actually learn release = f(target distance) instead of collapsing
to a constant throw. See release_bandit_env.py for the reformulation.

    python train_release_bandit.py --target-r 3.3 5.0 --steps 6000 --name bandit_v1
"""

import argparse
import csv
import json
import platform
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from release_bandit_env import ReleaseBanditEnv, make_bandit_env


def zero_initialise_actor(model):
    """Zero output layer -> the initial policy throws the nominal (centre)
    release, a sensible ~3.98 m starting point before learning to aim."""
    output = next(
        layer for layer in reversed(list(model.actor.mu.modules()))
        if isinstance(layer, torch.nn.Linear)
    )
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    model.actor_target.load_state_dict(model.actor.state_dict())


class BanditEvalCallback(BaseCallback):
    """Deterministically evaluate the aim across a distance sweep and keep the
    lowest-mean-miss controller."""

    def __init__(self, out_dir, env_kwargs, eval_distances,
                 eval_freq=2_000, checkpoint_freq=1_000, window=200):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.env_kwargs = dict(env_kwargs)
        self.eval_distances = tuple(float(d) for d in eval_distances)
        self.eval_freq = int(eval_freq)
        self.next_eval = self.eval_freq
        # Checkpoints are saved on their own (coarser) cadence so the fall-count-
        # by-version plot stays readable even when eval_freq is small.
        self.checkpoint_freq = int(checkpoint_freq)
        self.next_checkpoint = self.checkpoint_freq
        self.best_miss = np.inf
        self.recent_miss = deque(maxlen=window)
        self.recent_reward = deque(maxlen=window)
        self._eval_envs = None
        # metrics.csv drives the analysis plots; header written once at start.
        self.csv_path = self.out_dir / "metrics.csv"
        self._csv_header = (
            ["timestep", "rolling_reward", "mean_miss",
             "hit_rate_0.50m", "fall_rate"]
            + [f"miss_r{d:.2f}" for d in self.eval_distances])

    def _sweep_envs(self):
        if self._eval_envs is None:
            self._eval_envs = []
            for distance in self.eval_distances:
                kwargs = dict(self.env_kwargs)
                kwargs["fixed_target"] = (distance, 0.0)
                self._eval_envs.append(
                    (distance, ReleaseBanditEnv(**kwargs)))
        return self._eval_envs

    def _evaluate(self):
        misses, falls, throws = [], [], []
        for distance, env in self._sweep_envs():
            obs, _ = env.reset(seed=777)
            action, _ = self.model.predict(
                obs[None].astype(np.float32), deterministic=True)
            _, _, _, _, info = env.step(action[0])
            misses.append(info["objective_miss"])
            falls.append(float(info["fallen"]))
            throws.append(info["throw_distance"])
        result = {
            "mean_miss": float(np.mean(misses)),
            "fall_rate": float(np.mean(falls)),
            "hit_rate_0.50m": float(np.mean(np.asarray(misses) < 0.5)),
        }
        for distance, miss, throw in zip(
                self.eval_distances, misses, throws):
            self.logger.record(f"bandit_eval/miss_r{distance:.1f}", miss)
            self.logger.record(f"bandit_eval/threw_r{distance:.1f}", throw)
        self.logger.record("bandit_eval/mean_miss", result["mean_miss"])
        self.logger.record("bandit_eval/fall_rate", result["fall_rate"])
        self.logger.record(
            "bandit_eval/hit_rate_0.50m", result["hit_rate_0.50m"])
        return result, misses, throws

    def _on_training_start(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(self._csv_header)

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode_objective_miss" in info:
                self.recent_miss.append(float(info["episode_objective_miss"]))
        for reward in self.locals.get("rewards", []):
            self.recent_reward.append(float(reward))
        if self.recent_miss:
            self.logger.record(
                "bandit/rolling_mean_miss", float(np.mean(self.recent_miss)))
        if self.recent_reward:
            self.logger.record(
                "bandit/rolling_reward", float(np.mean(self.recent_reward)))

        while self.num_timesteps >= self.next_eval:
            result, misses, throws = self._evaluate()
            self.next_eval += self.eval_freq
            rolling_reward = (
                float(np.mean(self.recent_reward))
                if self.recent_reward else float("nan"))
            # One CSV row per eval: feeds the reward + success-rate curves.
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [self.num_timesteps, rolling_reward,
                     result["mean_miss"], result["hit_rate_0.50m"],
                     result["fall_rate"]] + [float(m) for m in misses])
            if result["mean_miss"] < self.best_miss:
                self.best_miss = result["mean_miss"]
                self.model.save(str(self.out_dir / "best_model"))
                payload = dict(result)
                payload["per_distance_miss"] = {
                    f"{d:.2f}": m
                    for d, m in zip(self.eval_distances, misses)}
                payload["per_distance_throw"] = {
                    f"{d:.2f}": t
                    for d, t in zip(self.eval_distances, throws)}
                payload["source"] = "deterministic sweep"
                (self.out_dir / "best_metrics.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8")
                self.logger.record("bandit_eval/best_mean_miss", self.best_miss)

        # Save "policy versions" on their own coarser cadence for the
        # fall-count-by-version plot.
        while self.num_timesteps >= self.next_checkpoint:
            self.model.save(
                str(self.out_dir / f"ckpt_{self.next_checkpoint}"))
            self.next_checkpoint += self.checkpoint_freq
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-r", nargs=2, type=float, default=[3.3, 5.0],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--target-theta", nargs=2, type=float,
                        default=[0.0, 0.0], metavar=("MIN", "MAX"))
    # Calibrated for the 4 cm ball: action[-1,1] spans ~3.4-5.5 m stable
    # landings (with a 4.38 m fall pocket the policy learns to avoid). The 2 cm
    # ball used 0.7776 / 0.0056.
    parser.add_argument("--release-window", type=float, default=0.0055)
    parser.add_argument("--release-center-offset", type=float, default=0.781)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--name", default="release_bandit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"],
                        default="subproc")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.4,
                        help="release exploration noise std")
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--eval-distances", nargs="*", type=float,
                        default=[3.3, 3.7, 4.1, 4.5, 5.0])
    parser.add_argument("--eval-grid", type=int, default=0,
                        help="if >0, evaluate on this many evenly-spaced "
                             "distances across target-r (a finer success-rate "
                             "signal than the default 5-point sweep)")
    args = parser.parse_args()
    if args.eval_grid > 0:
        args.eval_distances = list(np.linspace(
            args.target_r[0], args.target_r[1], args.eval_grid))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    torch.set_float32_matmul_precision("high")

    out = Path(__file__).resolve().parent / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "release_window": args.release_window,
        "release_center_offset": args.release_center_offset,
    }
    (out / "config.json").write_text(json.dumps({
        "algorithm": "TD3-bandit",
        "task": "one-step release-timing aim (walk+throw rolled out per step)",
        "target_r": args.target_r,
        "target_theta": args.target_theta,
        "release_window": args.release_window,
        "release_center_offset": args.release_center_offset,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "sigma": args.sigma,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }, indent=2), encoding="utf-8")

    env_fns = [make_bandit_env(rank, seed=args.seed, **env_kwargs)
               for rank in range(args.num_envs)]
    if args.vec_env == "subproc" and args.num_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)

    noise = NormalActionNoise(
        np.zeros(1, dtype=np.float32),
        np.full(1, args.sigma, dtype=np.float32))

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=200_000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,  # one-step episodes -> no bootstrapping, gamma is inert.
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(out / "tb"),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    zero_initialise_actor(model)

    print(f"training device: {args.device}; {args.num_envs} envs")
    callback = BanditEvalCallback(
        out, env_kwargs, args.eval_distances, eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq)
    try:
        model.learn(total_timesteps=args.steps, callback=callback,
                    progress_bar=True)
    finally:
        model.save(str(out / "final"))
        env.close()
        print(f"saved to {out}")


if __name__ == "__main__":
    main()

# G1 Humanoid — Learned Throwing & Target Aiming

**A Unitree G1 humanoid that walks and throws a ball to a target, learning *when to release* to hit a distance that changes every episode — generalizing across a 3.3–5.0 m band with a mean miss of 0.18 m and a 100 % hit rate within 0.5 m, and never falling.**

This bundle is the runnable "final version" of the project. It contains two trained controllers and everything needed to view them in the MuJoCo physics simulator.

| Controller | Task | What it demonstrates |
|---|---|---|
| **`bandit_bigball`** | Walk forward and throw at a **changing target** | *Aiming*: one policy that picks the release timing to hit any distance in 3.3–5.0 m |
| **`td3_throw_human`** | **Standing throw learned from scratch** (pelvis pinned) | *Pure RL*: discovers the whole throwing swing from a landing-error reward alone — **no scripted motion** — with a human-range elbow (no hyperextension) |

---

## 1. The problem

Manipulation robots are mostly *static* and *reach-bound*: to place an object they must travel to it, stop, place, and retract. Throwing removes both limits at once — it decouples *where the robot is* from *where the object goes*, and it never requires stopping the motion. The hard, differentiated part is doing it **on legs, while moving, and hitting a target at an arbitrary distance**.

The skill this project isolates is therefore not "throwing" but **learned release timing**: choosing the exact instant to let go so that a ballistic payload lands on a target, executed mid-gait without halting the body. That primitive is the building block for dynamic humanoid manipulation — tossing-to-place in logistics, tool hand-offs — and, as the next section argues, for *standoff delivery* in environments too dangerous for humans or conventional robots.

## 2. Motivation — standoff delivery in hazardous environments

The clearest near-term value of "throw a payload accurately while moving" is in places a robot must enter but should leave as quickly — and stay as far from the danger — as possible. **Nuclear decommissioning is the sharpest example.**

In a damaged reactor or a hot cell, the binding constraint is **cumulative radiation dose**, which is roughly `dose ≈ time × dose-rate`, and for a point source the dose-rate falls with the **square of the distance** (`∝ 1/r²`). This is not academic: at **Fukushima Daiichi**, multiple survey robots were disabled by radiation before completing their missions; at **Chernobyl**, remote machines failed outright and humans were sent in for seconds at a time.

Conventional robots make the dose problem *worse* in two ways:
- To place each sensor they **approach, stop, set it down, and retract** — the worst possible motion profile: maximum time in the field, minimum standoff distance.
- **Contamination is one-way.** Anything that crosses into a hot cell may never be allowed back out; a robot that physically visits every corner risks becoming radioactive waste itself.

A legged humanoid that **throws sensors into place while walking** inverts every one of those:

| Capability (what this project built) | Value in a hazardous cell |
|---|---|
| Release a payload accurately **from a distance** | Greater standoff → lower dose-rate on the robot's own electronics (`1/r²`) |
| Throw **without stopping** the gait | Less dwell time → less total dose; seed a whole grid in one continuous pass |
| Deliver **there** while staying **here** | Robot never enters/contaminates the target zone; can reach across gaps it cannot walk over |
| Aim across a **range of distances** (3.3–5.0 m) | Places sensors at a grid of survey points, each at a different range |
| One-way delivery | A *feature*, not a bug: in decommissioning you *want* to leave dosimeters behind |

That fourth row is exactly the capability demonstrated by `bandit_bigball`: aiming a payload to a target whose distance changes, mid-gait — i.e. seeding a room of survey points in a single walking pass.

The same primitive reskins to other standoff-delivery hazards — **CBRN** response, **firefighting** (placing beacons or suppressant charges in structures too unstable to enter), **EOD / demining**, and **planetary** instrument deployment. Nuclear leads because the benefit is quantifiable in the exact unit regulators already track: **radiation dose (Sieverts) avoided.**

**Honest scope.** Throwing *reduces* exposure and contamination risk; it does not remove the need to radiation-harden the platform, and everything here is a simulation result today. The contribution is the *dynamic-aiming primitive* that makes such a robot possible — not a fielded system.

## 3. The core technical contribution

Naïve multi-step reinforcement learning **fails** on this task, and understanding why is the crux of the work.

The controller runs at **50 Hz** and an episode is ~5 s, so a rollout is **~250 control steps**. The *release* decision matters at essentially **one** of those steps. When the release action is one channel of a 24-D action applied at every step, its learning signal is diluted by a factor of ~250. Empirically, TD3 collapses to a **single, constant, target-independent throw** (landing ~2 m regardless of the target) because "throw a safe nominal distance and stay upright" out-scores "aim well" under the diluted gradient.

**The fix — a one-step contextual bandit** (`release_bandit_env.py`). We roll the *entire* walk-and-throw out **inside a single environment step**, so every RL transition is exactly:

```
(target distance)  ->  (release-timing action)  ->  (landing reward)
```

with no temporal dilution. The agent then learns `release = f(target)` directly. Concretely:

- **Observation** (1-D): `distance − 4.0`, the target distance centred on the 4 m nominal.
- **Action** (1-D): release timing ∈ [−1, 1].
- **Reward**: `−landing_miss − 5·(fell)`.

During the bandit's inner rollout the body throws the **fixed nominal motion** (`stage="release"` in the env freezes all 23 non-release action channels), so exploration can never destabilise the robot — only the release instant varies. This is what turns an unstable multi-step credit-assignment problem into a clean, low-variance 1-D regression that TD3 solves reliably.

## 4. System architecture

```
          target distance (1-D context)
                     │
                     ▼
        ┌──────────────────────────┐
        │  TD3 bandit policy         │   ← learns release = f(distance)
        │  (bandit_bigball)          │
        └──────────────────────────┘
                     │  release timing ∈ [-1,1]
                     ▼
   ┌───────────────────────────────────────────┐
   │  WalkingThrowEnv  (MuJoCo, 50 Hz)           │
   │                                             │
   │  • Unitree G1 (37 DoF)                       │
   │  • Legs driven by a pretrained 50 Hz LSTM    │  ← pretrained/g1_walk_policy_lstm.pt
   │    walking policy (fixed)                     │
   │  • Upper body throws a fixed nominal swing    │
   │  • Ball held by a weld ("grip") released at   │
   │    the scheduled instant                      │
   └───────────────────────────────────────────┘
                     │
                     ▼
            ballistic flight → landing → miss
```

- **Physics**: MuJoCo, `timestep = 0.002 s`, `decimation = 10` → **50 Hz** control.
- **Robot**: Unitree G1 from MuJoCo Menagerie (`mujoco_menagerie/unitree_g1/`).
- **Locomotion**: a small pretrained LSTM walk policy (`pretrained/g1_walk_policy_lstm.pt`) runs the legs; RL never has to relearn walking, it composes a throw on top of a stable gait.
- **Release scheduling**: the nominal release time is `throw_start + release_center_offset`; the policy's action shifts it by `± release_window`:

  ```
  scheduled_release = throw_start + release_center_offset + release_window · action
  ```

  For `bandit_bigball`: `release_center_offset = 0.781 s`, `release_window = 0.0055 s` — calibrated so that action ∈ [−1, 1] maps smoothly onto the whole 3.3–5.0 m landing band.

### `td3_throw_human` — the standing throw learned from scratch (pure RL)
`td3_throw_human` is a pinned-base standing throw (`env.py: ThrowEnv`, `freeze_legs=True` on `models/throw_scene.xml`) trained **from scratch with no scripted swing** (`train_td3.py --no-warmstart`): a zero policy does not throw at all — the ball just drops at the robot's feet — so RL must *discover* the entire windup-and-swing purely from the landing-error reward (potential-shaped during flight, replaced by the true landing error at episode end; there are no pose, energy, survival, or bonus terms). It adds one constraint that a reward-only throw otherwise exploits: the **elbow cannot hyperextend** (`--elbow-min 0.0` clamps the elbow lower limit to 0 rad = straight), so the learned motion stays in a human range instead of bending backwards past straight. Trained with TD3 for 400 k steps; the checkpoint promoted to `best_model` (200 k steps) lands **0.35 m from a 4 m target, 100 % within 0.5 m** (deterministic, seed-independent). `runs/td3_throw_human/` keeps the full checkpoint history and `config.json`; a companion run `runs/td3_throw_human_fine/` (checkpoints every 25 k) backs the learning-progression video in `out_videos/`.

## 5. Results

`bandit_bigball`, deterministic sweep over 20 distances from 3.30 m to 5.00 m (`runs/bandit_bigball/best_metrics.json`):

| Metric | Value |
|---|---|
| Mean landing miss | **0.177 m** |
| Hit rate within 0.5 m | **100 %** |
| Fall rate | **0 %** |
| Per-distance miss range | 0.036 – 0.298 m |

**Training configuration** (`runs/bandit_bigball/config.json`): TD3 bandit · learning rate 1e-3 · batch 256 · action noise σ = 0.4 · 8 parallel envs · target range 3.3–5.0 m · seed 0 · PyTorch 2.11 (CUDA 12.8) · Python 3.13.

**Honest note on resolution.** Because the release window is small and release is quantised to physics sub-steps, the controller maps the continuous distance onto a handful of discrete landing bands (~3.42, 3.99, 4.38, 4.88 m). This is *coarse* aiming — but it still lands within 0.5 m across the entire 3.3–5.0 m range, which is the task tolerance. Finer resolution would come from a wider window plus sub-step release interpolation.

`td3_throw_human`, deterministic throw at a 4 m target (`runs/td3_throw_human/config.json`, TD3 · from scratch · elbow_min = 0.0 · seed 0):

| Metric | Value |
|---|---|
| Landing point | (4.32, 0.13) m |
| Landing miss | **0.348 m** (100 % within 0.5 m) |
| Elbow range | never hyperextends (≥ 0 rad = straight) |

Its value is as much qualitative as quantitative: `out_videos/throw_human_learning_timelapse.mp4` shows the honest learning arc — ball-at-feet before training, the swing emerging over 25 k–100 k, the first hit near 125 k, the best throw (0.23 m) at 375 k, and the real **non-monotonic TD3 wobble** in between (175 k and 275 k regress). Nothing is scripted; the swing is entirely RL-discovered.

## 6. Repository layout

```
g1-throwing/                          # repository root
├── README.md
├── requirements.txt
├── train/
│   ├── view_bandit.py                # viewer for the changing-target aiming sim
│   ├── train_release_bandit.py       # trains the bandit (release = f(distance))
│   ├── release_bandit_env.py         # 1-step contextual-bandit wrapper (the key idea)
│   ├── walking_throw_env.py          # MuJoCo walk+throw environment (24-D action, 50 Hz)
│   ├── walk_policy.py                # loads the pretrained LSTM walking policy
│   ├── env.py                        # ThrowEnv: pinned-base throw (supports --elbow-min)
│   ├── gym_env.py                    # Gym wrapper for ThrowEnv
│   ├── train_td3.py                  # trains the from-scratch standing throw (--no-warmstart, --elbow-min)
│   ├── eval_td3.py                   # viewer/eval for the standing throw (td3_throw_human)
│   ├── plot_throw_ablation.py        # top-down ablation: no-RL (ball drops) vs RL (learned throw)
│   ├── record_learning_timelapse.py  # renders the from-scratch learning timelapse
│   ├── out_videos/
│   │   └── throw_human_learning_timelapse.mp4   # RL discovering the throw, 25k→400k
│   └── runs/
│       ├── bandit_bigball/           # best_model.zip + config.json (aiming)
│       ├── td3_throw_human/          # best_model + vecnorm + config.json + checkpoints (pure-RL throw)
│       └── td3_throw_human_fine/     # 25k-spaced checkpoints backing the timelapse
├── models/
│   ├── walk_throw_scene.xml          # scene for the walk+throw task
│   └── throw_scene.xml               # scene for the standing throw
├── mujoco_menagerie/unitree_g1/      # G1 robot: g1.xml + assets/ (51 meshes)
└── pretrained/
    └── g1_walk_policy_lstm.pt        # frozen 50 Hz LSTM walking policy
```

All scripts resolve assets by **relative path from the repo root**, so this directory structure must be preserved.

## 7. Installation

Requires Python 3.13. Install dependencies:

```bash
pip install -r requirements.txt
# or: pip install mujoco stable-baselines3 torch gymnasium numpy
```

(CPU is sufficient to *view* the policies; a GPU was used only for training.)

## 8. Running the simulations

Run from inside `train/`.

### Changing-target aiming (`bandit_bigball`)
```bash
# RL policy aims at a sweep of distances
python view_bandit.py --run bandit_bigball --mode rl --targets 3.3 3.7 4.1 4.5 5.0

# Side-by-side: no-RL baseline (misses, ~4 m every time) then RL (aims)
python view_bandit.py --run bandit_bigball --mode compare

# No-RL baseline only (shows it cannot aim)
python view_bandit.py --run bandit_bigball --mode baseline
```
After each throw the viewer prints, e.g.:
`[RL policy]  target=4.50 m  ->  threw to 4.47 m  (HIT by 0.03 m)`

Retrain it yourself (exact hyperparameters are recorded in `runs/bandit_bigball/config.json`; run with `--help` for all flags):
```bash
python train_release_bandit.py --name bandit_bigball --steps 6000
```

### Pure-RL human throw (`td3_throw_human`)
```bash
# Watch the from-scratch, human-elbow throw (elbow clamp applied automatically from config)
python eval_td3.py --run td3_throw_human --view

# Quantitative evaluation (deterministic: lands ~0.35 m from 4 m, 100% within 0.5 m)
python eval_td3.py --run td3_throw_human --episodes 5

# Top-down ablation figure: no-RL (ball drops at feet) vs RL (learned throw)
python plot_throw_ablation.py --run td3_throw_human

# Re-render the learning timelapse (needs: pip install imageio imageio-ffmpeg Pillow)
python record_learning_timelapse.py --run td3_throw_human_fine --out out_videos/throw_human_learning_timelapse.mp4
```
Retrain it from scratch yourself:
```bash
python train_td3.py --no-warmstart --target 4 0 --elbow-min 0.0 --name td3_throw_human --steps 400000
```

## 9. Known limitations

- **Simulation only.** These are MuJoCo results; sim-to-real on hardware (and radiation-hardening for the hazardous-environment use case) is future work.
- **Coarse aiming resolution** — see the note in §5.
- **`td3_throw_human` is a pinned, arm-only throw** (pelvis welded, legs frozen) — it demonstrates RL discovering an accurate throwing swing, not a full-body balanced throw.
- **Physics engine.** Behaviour is calibrated to MuJoCo; a straight port of the policy into a different engine (e.g. Unity PhysX) will not reproduce it without retraining.

## 10. Attribution

The Unitree G1 model (`mujoco_menagerie/unitree_g1/`) is from Google DeepMind's **MuJoCo Menagerie**; refer to that project for its model license. All training code, environments, and policies in this bundle are the project's own work.

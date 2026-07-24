# G1 Walking + Throwing (Level 3)

This project enables the Unitree G1 to throw a ball while continuously walking forward:

1. The robot first establishes a stable walking gait for approximately 1.4 seconds.
2. It maintains positive forward velocity throughout the throwing motion.
3. After releasing the ball, it continues walking in a straight line.
4. A pretrained LSTM policy controls the 12 leg joints.
5. TD3 learns residual actions for throwing, release timing, and whole-body balance.
6. The pelvis is not fixed.

The walking policy is based on
[HusseinLezzaik/unitree-g1-bipedal-rl-walk](https://github.com/HusseinLezzaik/unitree-g1-bipedal-rl-walk).
See `THIRD_PARTY_NOTICES.md` for license information.

## Verified Default Results

The default target is `(4, 0)`, the commanded forward velocity is `1.3 m/s`, and each episode lasts 5 seconds.

Results from a 10-episode evaluation of `td3_walk_throw_4_0_vx1p3/best_model.zip`:

| Metric | Result |
|---|---:|
| Mean landing error | 0.051 m |
| Hit rate within 0.50 m | 100% |
| Distance traveled before the throw | 0.707 m |
| Distance traveled during the throw | 0.881 m |
| Distance traveled after the throw | 2.555 m |
| Positive-motion fraction during the throw | 100% |
| Minimum forward velocity during the throw | 0.383 m/s |
| Total forward distance | 4.143 m |
| Final lateral drift | 0.002 m |
| Maximum line-tracking error | 0.166 m |
| Fall rate | 0% |

## Quick Check

```powershell
cd C:\Users\25954\Documents\ADC_Project\basic-test\g1-throwing-walking
C:\Users\25954\anaconda3\envs\mujo_torch\python.exe train\check_walk_integration.py
```

## Evaluate the Trained Policy

Run a 50-episode evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval.ps1 `
  --run td3_walk_throw_4_0_vx1p3 --episodes 50
```

Open the MuJoCo viewer:

```powershell
powershell -ExecutionPolicy Bypass -File .\eval.ps1 `
  --run td3_walk_throw_4_0_vx1p3 --view --cam follow
```

The evaluation output reports:

- `before throw`: forward distance traveled before the throwing motion starts;
- `during throw`: forward distance traveled throughout the complete throwing motion;
- `after throw`: forward distance traveled after the throwing motion ends;
- `moving in throw`: fraction of the throwing period with positive forward velocity;
- `min throw vx`: minimum forward velocity during the throwing period;
- `max line error`: maximum lateral deviation from the target line.

## Training on an RTX 4060 GPU

In the first stage, train only the right arm and release timing:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage throw --steps 500000 `
  --name td3_walk_throw_4_0_vx1p3 `
  --resume train\runs\td3_walk_throw_4_0_vx1p3\best_model.zip
```

After the first stage becomes stable, unlock the leg, waist, and left-arm balance residuals:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage full --steps 1000000 `
  --name td3_walk_throw_4_0_vx1p3 `
  --resume train\runs\td3_walk_throw_4_0_vx1p3\best_model.zip
```

Default training configuration:

- The TD3 networks run on CUDA.
- Four MuJoCo worker processes run in parallel.
- MuJoCo runs at 500 Hz and the control policy at 50 Hz.
- `--learning-starts 0` prevents full-range random actions from disrupting the gait.
- The learning rate is `1e-5` for conservative fine-tuning from the strong warm start.
- `best_model.zip` is overwritten only when the aggregate evaluation metrics genuinely improve.

TensorBoard:

```powershell
C:\Users\25954\anaconda3\envs\mujo_torch\Scripts\tensorboard.exe `
  --logdir train\runs
```

Key metrics to monitor:

- `level3/mean_miss`
- `level3/fall_rate`
- `level3/mean_distance_during_throw`
- `level3/mean_max_abs_lateral_error`
- `level3_eval/positive_velocity_fraction_during_throw`

## Control Architecture

TD3 produces a 24-dimensional residual action:

| Range | Function |
|---|---|
| 0–11 | Residuals for the 12 leg joints |
| 12–14 | Waist yaw/roll/pitch residuals |
| 15–18 | Left-arm residuals |
| 19–22 | Right-arm residuals |
| 23 | Release-timing residual |

The final leg-joint target is:

```text
pretrained_walk_target + 0.06 rad * td3_leg_residual
```

The walking policy's velocity command is also corrected by closed-loop lateral-position and heading feedback, guiding the robot back toward the forward line established at reset.

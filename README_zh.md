# G1 Walking + Throwing（Level 3）

这个项目实现 Unitree G1 在持续向前行走的过程中投球：

1. 机器人先稳定行走约 1.4 秒；
2. 投球动作期间继续保持正向速度；
3. 球释放后继续沿直线行走；
4. 预训练 LSTM 控制 12 个腿关节；
5. TD3 学习投球、释放时刻和全身平衡 residual；
6. 骨盆不固定。

行走策略来自
[HusseinLezzaik/unitree-g1-bipedal-rl-walk](https://github.com/HusseinLezzaik/unitree-g1-bipedal-rl-walk)，
许可证信息见 `THIRD_PARTY_NOTICES.md`。

## 已验证的默认结果

默认目标为 `(4, 0)`，前进指令为 `1.3 m/s`，每回合 5 秒。

`td3_walk_throw_4_0_vx1p3/best_model.zip` 的 10 回合测试结果：

| 指标 | 结果 |
|---|---:|
| 平均落点误差 | 0.051 m |
| 0.50 m 内命中率 | 100% |
| 投球前前进距离 | 0.707 m |
| 投球中前进距离 | 0.881 m |
| 投球后前进距离 | 2.555 m |
| 投球期间正向运动比例 | 100% |
| 投球期间最低前向速度 | 0.383 m/s |
| 总前进距离 | 4.143 m |
| 最终横向漂移 | 0.002 m |
| 最大直线误差 | 0.166 m |
| 摔倒率 | 0% |

## 快速检查

```powershell
cd C:\Users\25954\Documents\ADC_Project\basic-test\g1-throwing-walking
C:\Users\25954\anaconda3\envs\mujo_torch\python.exe train\check_walk_integration.py
```

## 查看训练结果

统计 50 回合：

```powershell
powershell -ExecutionPolicy Bypass -File .\eval.ps1 `
  --run td3_walk_throw_4_0_vx1p3 --episodes 50
```

打开 MuJoCo 动画：

```powershell
powershell -ExecutionPolicy Bypass -File .\eval.ps1 `
  --run td3_walk_throw_4_0_vx1p3 --view --cam follow
```

评估输出会分别显示：

- `before throw`：投球开始前的前进距离；
- `during throw`：完整投球动作期间的前进距离；
- `after throw`：投球结束后的前进距离；
- `moving in throw`：投球期间正向速度所占比例；
- `min throw vx`：投球期间最低前向速度；
- `max line error`：相对目标直线的最大横向误差。

## RTX 4060 GPU 训练

第一阶段只训练右臂和释放时刻：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage throw --steps 500000 `
  --name td3_walk_throw_4_0_vx1p3 `
  --resume train\runs\td3_walk_throw_4_0_vx1p3\best_model.zip
```

第一阶段稳定后，解锁腿部、腰部和左臂平衡 residual：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage full --steps 1000000 `
  --name td3_walk_throw_4_0_vx1p3 `
  --resume train\runs\td3_walk_throw_4_0_vx1p3\best_model.zip
```

训练默认配置：

- TD3 网络使用 CUDA；
- 4 个并行 MuJoCo 子进程；
- MuJoCo 500 Hz、控制策略 50 Hz；
- `--learning-starts 0`，避免全范围随机动作破坏步态；
- 学习率 `1e-5`，对强 warm-start 做保守微调；
- `best_model.zip` 只有综合指标真正改善时才会覆盖。

TensorBoard：

```powershell
C:\Users\25954\anaconda3\envs\mujo_torch\Scripts\tensorboard.exe `
  --logdir train\runs
```

重点观察：

- `level3/mean_miss`
- `level3/fall_rate`
- `level3/mean_distance_during_throw`
- `level3/mean_max_abs_lateral_error`
- `level3_eval/positive_velocity_fraction_during_throw`

## 控制结构

TD3 使用 24 维 residual 动作：

| 范围 | 内容 |
|---|---|
| 0–11 | 12 个腿关节 residual |
| 12–14 | 腰部 yaw/roll/pitch residual |
| 15–18 | 左臂 residual |
| 19–22 | 右臂 residual |
| 23 | 释放时间 residual |

腿部最终目标：

```text
pretrained_walk_target + 0.06 rad * td3_leg_residual
```

行走策略的速度指令还会经过横向位置和航向闭环修正，使机器人回到
reset 时确定的前向直线上。

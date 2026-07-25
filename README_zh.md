# G1 Walking + Throwing（Level 3）

这个项目实现 Unitree G1 在持续向前行走的过程中投球：

1. 机器人先稳定行走约 1.4 秒；
2. 投球动作期间继续保持正向速度；
3. 球释放后继续沿直线行走；
4. 预训练 LSTM 控制 12 个腿关节；
5. 项目同时保留旧版校准演示，并新增一条不使用预设投球轨迹的 TD3
   铅球式训练路线；
6. 骨盆不固定。

行走策略来自
[HusseinLezzaik/unitree-g1-bipedal-rl-walk](https://github.com/HusseinLezzaik/unitree-g1-bipedal-rl-walk)，
许可证信息见 `THIRD_PARTY_NOTICES.md`。

> 以下结果属于旧版校准演示，不能作为 TD3 已经学会 Level 3 组合动作的
> 证明。真正的强化学习请使用本文后面的“真正学习型铅球式投掷”路线。

## 已验证的旧版演示结果

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

## 真正学习型铅球式投掷

新路线与旧版校准演示完全分开：

| 路线 | 启动脚本 | 投球动作来源 |
|---|---|---|
| 旧版演示 | `train_gpu.ps1` | 校准好的手臂轨迹叠加 TD3 residual |
| 学习型铅球 | `train_shotput_gpu.ps1` | TD3 直接控制，不存在预设投球轨迹 |

在学习型路线中，未训练的零输出 actor 只会保持手臂中立，不会执行隐藏的
蓄力或释放动作。奖励引导策略自己发现以下动作顺序：

1. 把右手抬到颈部右侧的准备区域；
2. 保持肘部抬高；
3. 朝目标方向向前推出手臂；
4. 伸展手臂并选择释放时刻；
5. 同时保持向前行走、直线跟踪和身体平衡。

准备姿态只作为奖励目标使用，代码不会把它直接写入机器人控制量。

### RTX 4060 分阶段训练

第一阶段从零学习右臂和释放时刻：

```powershell
cd C:\Users\25954\Documents\ADC_Project\basic-test\g1-throwing-walking
powershell -ExecutionPolicy Bypass -File .\train_shotput_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage arm --steps 300000 `
  --name td3_shotput_walk_4_0_vx1p3
```

第二阶段解锁腰部和左臂进行反向平衡：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_shotput_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage upper --steps 700000 --learning-starts 0 `
  --name td3_shotput_walk_4_0_vx1p3 `
  --resume train\runs\td3_shotput_walk_4_0_vx1p3\best_model.zip
```

第三阶段解锁小幅腿部 residual，学习全身平衡：

```powershell
powershell -ExecutionPolicy Bypass -File .\train_shotput_gpu.ps1 `
  --target 4 0 --command 1.3 0 0 `
  --stage full --steps 1000000 --learning-starts 0 `
  --name td3_shotput_walk_4_0_vx1p3 `
  --resume train\runs\td3_shotput_walk_4_0_vx1p3\best_model.zip
```

如果还没有生成 `best_model.zip`，不要进入下一阶段。新路线只把未训练模型
保存为 `initial_model.zip`；只有确定性评估同时满足以下条件，才会生成真正
的 `best_model.zip`：

- 综合分数优于零 actor；
- 至少 60% 的评估回合中球成功落地；
- 平均铅球动作风格分数不低于 0.35；
- 至少 40% 的回合通过完整动作风格检查；
- 摔倒率不高于 20%。

评估通过门槛的模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\eval_shotput.ps1 `
  --run td3_shotput_walk_4_0_vx1p3 --episodes 50
```

在 MuJoCo 中查看：

```powershell
powershell -ExecutionPolicy Bypass -File .\eval_shotput.ps1 `
  --run td3_shotput_walk_4_0_vx1p3 --view --cam follow
```

`--ckpt final` 只用于检查尚未完成的训练。评估器会输出落点误差、准备姿态
手部误差、释放手速/高度、肘部角度、动作风格有效率、行走距离、直线误差
和摔倒率。

每个训练目录都会生成明确的学习审计文件：

- `initial_metrics.json`：未训练零 actor 的确定性基准；
- `final_metrics.json`：本次训练结束后的确定性结果；
- `best_metrics.json`：只有学习策略通过所有门槛时才存在；
- `learning_proof.json`：初始/最终对照以及 actor 参数变化。

TensorBoard 中的新指标使用 `shotput/` 和 `shotput_eval/` 前缀。项目内的
`td3_shotput_smoke` 仅用于验证 CUDA 训练和文件保存链路，不是最终策略。

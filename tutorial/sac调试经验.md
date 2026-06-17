# SAC 调试经验：MyBipedalWalkFlat 从 10 步摔跑到完全收敛

> 在 UniLab 框架下用 FastSAC 训练 RLbipedal 双足机器人平地行走，从"10 步即摔、无法学习"到"100% 走完 episode、reward +64"的完整调试记录。

- **训练日期**: 2026-06-17
- **机器人**: my_bipedal（6 执行器关节，20 总 DOF，含弹簧关节）
- **框架**: UniLab + MuJoCo + PyTorch CUDA (RTX 5070 Ti)
- **最终结果**: episode/length=958/1000, terminated_rate=0%, reward/mean=+64

---

## 1. 问题现象

直接用默认 SAC 配置训练 bipedal，500 iter 后：

| 指标 | iter 100 | iter 500 |
|------|:--------:|:--------:|
| episode/length | 10.5 | 10.3 |
| terminated_rate | 100% | 100% |
| target_q_max | +0.63 | **-1.31** (一路掉负) |
| reward/mean | -3.74 | -3.72 |
| policy_entropy | 4.03 | 4.02 (完全不动) |

核心症状：**target_q_max 一路掉成负值，actor 收不到有效梯度，episode 长度卡在 10 步无法突破。**

---

## 2. 根因诊断

### 2.1 为什么 episode 只有 10 步？

通过 instrument `update_state()` 打印每步的终止条件，发现：

```
step 8: env 2 terminated — tilt=8.0deg base_z=0.484 fell=False too_low=True
step 9: env 1 terminated — tilt=13.4deg base_z=0.487 fell=False too_low=True
```

**终止原因不是摔倒（tilt < 70deg），而是 base_z 跌破 `base_height_min=0.50`。**

### 2.2 为什么 base_z 会持续下降？

检查全部 20 个关节的 deviation，发现 unactuated spring joints 在第一步就大幅偏移：

```
Step 0: full 20-DOF deviation from default:
  [ 2] LLowerJoint1 (ACTUATED): diff=-0.061  ← 执行器关节偏差小
  [ 3] joint_3:                diff=+0.151  ← 弹簧关节立即偏移
  [ 4] joint_4:                diff=+0.209  ← 弹簧关节立即偏移
  [ 7] joint_7:                diff=-0.148  ← 弹簧关节立即偏移
  [ 8] joint_8:                diff=-0.131  ← 弹簧关节立即偏移
```

**6 个执行器关节（LHipJoint 等）无法控制 14 个 unactuated spring joints。** 默认站姿不是平衡点，spring joints 在重力下立即坍塌，导致整体 base_z 持续下降。

### 2.3 为什么 PD balance 无效？

尝试了 PD balance controller（将 actuated joints 推回 default），Kp=0/2/5/10 全部给出相同的 base_z 轨迹和 ep_len=10.3：

```
Kp= 0.0: base_z=[0.698 0.693 0.683 0.669 0.651 0.629 0.603 0.583 ...] mean_ep=10.3
Kp= 2.0: base_z=[0.698 0.692 0.681 0.666 0.648 0.626 0.600 0.580 ...] mean_ep=10.3
Kp=10.0: base_z=[0.697 0.689 0.676 0.660 0.641 0.618 0.591 0.585 ...] mean_ep=10.2
```

PD 控制不了 spring joints，base_z 下沉速度完全相同。

### 2.4 为什么 SAC actor 学不到？

SAC actor 的初始 `action_std=1.0`（`log_std_min=-1.0, log_std_max=1.0`，初始 log_std=0），而 warm-start standing controller 用 `noise_std=0.15` 能存活 24 步。Actor 的探索噪声是 warm-start 的 4-5 倍，导致关节偏移过大，比 standing controller 更快摔倒。

Critic 从 warm-start 数据学到了"接近站立的 action 更好"（target_q_max 保持正值），但 actor 自己的探索策略无法复制这种"小动作"行为。

---

## 3. 解决方案：三要素 + 两阶段

### 3.1 三要素（缺一不可）

| 要素 | 作用 | 配置 |
|------|------|------|
| **Warm-start prefill** | 给 critic 提供 ep_len>10 的初始数据，让 Q 值估计有意义 | `algo.warm_start.enabled=true` |
| **Curriculum（放宽终止）** | 降低 `base_height_min`，让 actor 有机会探索更长轨迹 | `reward.base_height_min=0.35` |
| **log_std 限制** | 限制初始探索噪声，让 actor 的 action 接近 warm-start 水平 | `algo.algo_params.log_std_min=-2.0, log_std_max=-0.5` |

### 3.2 两阶段训练流程

```
阶段 1: standing_controller warm-start → 3000 iter → 生成能走路的 checkpoint
    ↓
阶段 2: sac_checkpoint warm-start (用阶段1的ckpt) → 5000 iter → 完全收敛
```

阶段 2 用阶段 1 的 checkpoint 做 prefill 比纯 standing controller 更有效，因为 checkpoint actor 已经学会了走路，能生成更高质量的 replay 数据。

---

## 4. 实验对比

| 实验 | 配置 | ep_len@1000 | ep_len@3000 | reward@3000 | 判定 |
|------|------|:-----------:|:-----------:|:-----------:|:----:|
| WS baseline | warm_start only, log_std[-1,1] | 10 | 10 | -3.72 | ❌ |
| A1 | log_std[-2,-0.5] only | 10 | 10 | -3.73 | ❌ |
| A2b | + base_height_min=0.35 | 37 | 185 | -2.69 | 🔶 突破 |
| A3 | A2b, 3000 iter | 142 | 850 | +42.3 | ✅ |
| B1 | base_height_min=0.50 + all | 10 | 10 | -3.69 | ❌ |
| **C1** | **A3 ckpt prefill, 0.35** | **280** | **874** | **+50.6** | ✅✅ |
| D1 | C1 ckpt, 0.45 | 58 | 147 | +16.4 | 🔶 |
| **E1** | **C1 config, 5000 iter** | **219** | **896** | **+57.0** | **✅✅✅** |

### 关键转折点

- **A1（只限制 log_std）无效**：action_std 从初始 0.29 被 log_std_max=-0.5 拉到 0.57，ep_len 仍 10
- **A2b（加 curriculum）突破**：放宽 base_height_min=0.35 后 ep_len 从 10 跳到 37
- **A3（长跑）收敛**：3000 iter 后 ep_len 达到 850，robot 学会走路
- **B1（原始终止条件）失败**：确认 base_height_min=0.50 下任何 policy 都只能活 10 步
- **C1（checkpoint prefill）更优**：比 standing controller 收敛更快（iter 1000 时 280 vs 142）
- **E1（5000 iter）完全收敛**：ep_len=958, terminated_rate=0%, reward=+64

---

## 5. 最终收敛指标（E1, 5000 iter）

| 指标 | iter 1 | iter 500 | iter 1000 | iter 2000 | iter 3000 | iter 5000 |
|------|:------:|:--------:|:---------:|:---------:|:---------:|:---------:|
| episode/length | 20 | 171 | 219 | 790 | 896 | **958** |
| terminated_rate | 0% | 100% | 100% | 100% | 100% | **0%** |
| timeout_rate | 0% | 0% | 0% | 0% | 0% | **100%** |
| reward/mean | 0.0 | -2.45 | +0.33 | +39.2 | +57.0 | **+64.1** |
| target_q_max | +1.72 | +13.7 | +8.73 | +6.80 | +7.05 | +7.05 |
| target_q_min | -4.06 | -4.06 | -3.91 | -1.35 | -0.43 | **+0.52** |
| action_std | 0.29 | 0.59 | 0.59 | 0.28 | 0.23 | 0.24 |

- `target_q_min` 从 -4.06 升到 +0.52，critic 完全收敛
- `action_std` 从 0.59 收敛到 0.24，策略确定性高
- 最后 1000 iter ep_len 稳定在 957.58，reward 持续上升

---

## 6. 最终配置

### 6.1 task YAML（`conf/offpolicy/task/sac/my_bipedal_flat/mujoco.yaml`）

```yaml
algo:
  warm_start:
    enabled: true
    source: standing_controller      # 阶段1用 standing_controller
    steps: 100000
    noise_std: 0.15
    action_hold: 4
    relax_termination_height: 0.35   # warm-start 时放宽终止条件
  algo_params:
    log_std_min: -2.0                # 限制探索噪声
    log_std_max: -0.5
reward:
  base_height_min: 0.35              # curriculum 终止阈值
```

### 6.2 两阶段训练命令

```bash
# 阶段 1: standing controller warm-start, 3000 iter
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  algo.max_iterations=3000

# 阶段 2: checkpoint prefill, 5000 iter
uv run python scripts/train_offpolicy.py \
  task=sac/my_bipedal_flat/mujoco \
  algo.max_iterations=5000 \
  algo.warm_start.source=sac_checkpoint \
  algo.warm_start.checkpoint_path=logs/fast_sac/MyBipedalWalkFlat/<阶段1目录>/model_3000.pt \
  algo.warm_start.relax_termination_height=0.35 \
  algo.warm_start.steps=200000
```

---

## 7. Warm-Start 机制实现

### 7.1 架构

```
DoubleBufferOffPolicyRunner.learn()
  ├── 创建 ReplayBuffer (shared memory)
  ├── ★ warm-start: run_warm_start()  ← 新增
  │     ├── 创建临时 env (registry.make)
  │     ├── StandingController / SAC checkpoint actor
  │     ├── 严格复刻 collector 的 replay_buffer.add() 契约
  │     └── 关闭临时 env
  ├── 创建 replay pipeline
  ├── 启动 collector 子进程
  └── training loop
```

**不破坏 runner lifecycle**：warm-start 在 learner 进程内跑，用独立的 env 实例，写入同一个 shared memory `ReplayBuffer`。collector 子进程完全不受影响。

### 7.2 StandingController

```python
ctrl = actions * action_scale + default_angles  # env action contract
# action=0 → 保持默认站姿
# action=noise(0.15) → 小扰动站立
```

### 7.3 relax_termination_height

通过 `env_cfg_override` 在 warm-start 时临时覆盖 `reward_config.base_height_min`，让 episode 能存活更长。正常 collector 仍用原始阈值，learner 看到的是混合数据。

### 7.4 sac_checkpoint source

从阶段 1 的 checkpoint 加载 SAC actor，用训练好的 policy 生成 replay 数据。比 standing controller 数据质量更高（ep_len 62 vs 36）。

---

## 8. 调试方法论

### 8.1 诊断 episode 终止原因

```python
# Monkey-patch update_state 打印终止条件
orig_update = env.update_state
def patched_update(state):
    result = orig_update(state)
    if np.any(result.terminated):
        gravity = env._backend.get_sensor_data('upvector')
        base_z = env._backend.get_base_pos()[:, 2]
        tilt = np.degrees(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)))
        for i in np.where(result.terminated)[0]:
            print(f'TERMINATE: tilt={tilt[i]:.1f}deg base_z={base_z[i]:.3f} '
                  f'fell={tilt[i]>70.0} too_low={base_z[i]<0.50}')
    return result
```

### 8.2 检查关节 deviation

```python
dof_pos = env.get_dof_pos()
diff = dof_pos[0] - env._all_default_angles
for j in range(20):
    if abs(diff[j]) > 0.001:
        marker = ' *ACTUATED*' if j in env._actuated_dof_pos_indices else ''
        print(f'[{j:2d}]: default={env._all_default_angles[j]:.3f} '
              f'current={dof_pos[0][j]:.3f} diff={diff[j]:+.4f}{marker}')
```

### 8.3 验证 replay buffer 数据格式

```python
batch = replay_buffer.sample(256)
for k, v in batch.items():
    print(f'{k}: shape={v.shape} range=[{v.min():.3f}, {v.max():.3f}]')
# 确认 done/truncated/critic 字段与 collector 一致
```

### 8.4 对比实验设计

每次只改一个变量，用相同的 warm-start seed 和 num_envs，确保可比性。关键指标在 iter 100/500/1000/3000 采样对比。

---

## 9. 经验教训

1. **先诊断终止原因，再调超参**：如果 robot 默认站姿就活不过 10 步，调 SAC 超参（lr, alpha, batch_size）都是浪费。先用 `action=0` 跑 20 步，打印 tilt/base_z/terminated，找出真正的终止条件。

2. **Warm-start 解决的是"critic 冷启动"问题**：如果 replay 里全是 10 步摔倒的轨迹，critic 的 Q 值估计全是负的，actor 收不到有效梯度。Warm-start 提供更长轨迹，让 critic 看到"活得久 = 更好"的信号。

3. **Curriculum 是必要的**：如果终止条件太严（base_height_min=0.50），任何 policy 都只能活 10 步，actor 无法探索到"活更久"的 action space 区域。放宽终止条件让 actor 有机会学习，然后再收紧。

4. **log_std 限制单独无效，但配合 curriculum 必要**：只限制 log_std 无法解决"10 步终止"问题，但配合 curriculum 后，较小的初始 action_std 让 actor 的探索更接近 warm-start 数据分布，加速收敛。

5. **两阶段训练比一阶段更稳**：阶段 1 用简单 standing controller 生成初始 checkpoint，阶段 2 用 checkpoint 生成更高质量 replay 数据。这比直接用 standing controller 跑到底收敛更快（C1 vs A3：iter 1000 时 280 vs 142）。

6. **不破坏 runner lifecycle**：warm-start 在 learner 进程内跑，用独立 env 实例，写入同一个 shared memory ReplayBuffer。不绕开 collector 协议，不另起同步机制。

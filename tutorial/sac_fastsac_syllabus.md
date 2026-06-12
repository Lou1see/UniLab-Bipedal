# UniLab SAC / FastSAC 教学大纲

> **适用对象**：有一定 Python / PyTorch 基础，想系统学习 UniLab 中 SAC 实现的学习者。
> **学习目标**：能跑通训练、能读懂核心代码、能改配置做小实验、能对比 FastSAC 与 FlashSAC 的工程差异。

---

## 第 0 阶段 · 前置知识

在深入代码之前，建议先理解以下标准 SAC 概念（不需要看 UniLab 代码）：

| 概念 | 一句话说明 |
|------|-----------|
| **Actor** | 策略网络，输入 observation，输出 action |
| **Critic** | Q 值网络，输入 (obs, action)，输出 Q 值 |
| **Target Critic** | Critic 的慢速拷贝，稳定 TD target 计算 |
| **Entropy Temperature (α)** | 控制探索-利用权衡的系数 |
| **Replay Buffer** | 存储历史 transition，离线采样训练 |
| **Off-policy Update** | 用 buffer 中任意旧数据更新网络 |

---

## 第 1 阶段 · 先跑起来

**目标**：理解"配置如何进入训练脚本"，不是调好结果。

**核心文件**：

| 文件 | 作用 |
|------|------|
| `conf/offpolicy/config.yaml` | Hydra 入口配置，默认 `algo: sac` |
| `conf/offpolicy/algo/sac.yaml` | SAC 算法配置，`algo_log_name: fast_sac` |
| `scripts/train_offpolicy.py` | 统一 off-policy 训练入口 |

**关键链路**：

```text
conf/offpolicy/config.yaml  (defaults: algo=sac)
  └─> conf/offpolicy/algo/sac.yaml  (algo_log_name: fast_sac)
      └─> scripts/train_offpolicy.py 中 algo_name=="sac" 分支
          └─> FastSACLearner
```

**建议运行命令**：

```bash
uv run python scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco
```

**验收标准**：能解释 `algo=sac` 如何变成 `FastSACLearner`。

---

## 第 2 阶段 · 训练入口与组件组装

**目标**：理解 Hydra config → Learner → Runner → Env 的完整组装链路。

**重点读**：

| 位置 | 函数 / 变量 |
|------|------------|
| `scripts/train_offpolicy.py` | `build_runner()` |
| `scripts/train_offpolicy.py` | `play_offpolicy()` |

**学习问题清单**：

1. `cfg.algo` 从哪里来？（→ Hydra compose）
2. `env_cfg_override` 是怎么构造的？
3. `FastSACLearner.__init__` 需要哪些维度参数？
4. `obs_dim`、`critic_obs_dim`、`action_dim` 是在哪里确定的？
5. checkpoint 如何保存和加载？

**数据流链路**：

```text
Hydra config
  → scripts/train_offpolicy.py
  → build_runner()
  → FastSACLearner
  → DoubleBufferOffPolicyRunner
  → env + replay buffer + learner update loop
```

---

## 第 3 阶段 · FastSAC 网络结构

**目标**：读懂 Actor / Critic 的前向传播，理解分布 Q 网络。

**核心文件**：`src/unilab/algos/torch/fast_sac/learner.py`

**核心类**：

| 类 | 职责 |
|----|------|
| `SACActor` | 输入 obs → 输出 action / mean / log_std |
| `DistributionalQNetwork` | 分布 Q 网络（C51 风格） |
| `SACCritic` | 封装多个 Q 网络 |
| `FastSACLearner` | Learner 主类，聚合所有组件 |

**关键理解点**：

- `SACActor` 使用 **tanh-squashed Gaussian** 输出有界 action
- Critic 不是普通标量 Q，而是 **distributional Q**（C51）
- `num_atoms` 控制 Q 分布的离散桶数量（默认 51）
- `qnet` 和 `qnet_target` 的关系：target 是慢速拷贝

**建议先读这些函数**：

- `SACActor.forward()`
- `SACActor.get_actions_and_log_probs()`
- `SACCritic.forward()`
- `FastSACLearner.__init__()`

---

## 第 4 阶段 · SAC 更新公式在代码中的位置

**目标**：建立"论文公式 ↔ 代码变量"的手写映射。

**核心文件**：`src/unilab/algos/torch/fast_sac/learner.py`

**SAC 概念 → 代码位置映射表**：

| SAC 概念 | 代码位置 |
|----------|---------|
| replay batch 采样 | `update_critic(batch)` / `update_actor(batch)` |
| target action 采样 | `_get_actions_and_log_probs_for_critic()` |
| critic TD target 计算 | `_critic_loss_tensors()` |
| actor loss 计算 | `_actor_loss_tensors()` |
| entropy temperature α | `log_alpha`，`alpha_optimizer` |
| target network 软更新 | `soft_update_target()` |

**建议练习**：手写一份映射——

> SAC 公式中的 `r`、`γ`、`done`、`α`、`log π`、`Q` 分别对应代码里的哪个变量？

---

## 第 5 阶段 · Runner 与 Replay Buffer

**目标**：理解采样和训练是解耦的两个动作。

**核心文件**：

| 文件 | 职责 |
|------|------|
| `src/unilab/algos/torch/offpolicy/double_buffer_runner.py` | 主 Runner，协调采样和训练 |
| `src/unilab/algos/torch/offpolicy/worker.py` | 数据收集子进程 |
| `src/unilab/ipc/replay_buffer.py` | 基于共享内存的打包 Replay Buffer |

**关键概念**：

- 环境 step 后 transition 写入 ReplayBuffer（worker 进程）
- Learner 从 ReplayBuffer 采样 batch（主进程）
- 旧数据会被重复学习（off-policy 本质）

**关键参数**：`updates_per_step`、`learning_starts`、`batch_size`、`replay_buffer_n`、`env_steps_per_sync`

---

## 第 6 阶段 · 配置如何影响训练

**目标**：能看懂配置，能做单变量实验。

**核心配置**：`conf/offpolicy/algo/sac.yaml`

**关键参数速查**：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `num_envs` | 并行环境数 | 4096 |
| `batch_size` | 每次 update 的 batch 大小 | 8192 |
| `replay_buffer_n` | buffer 容量 | 3_000_000 |
| `updates_per_step` | 每步更新的次数 | 1 |
| `learning_starts` | 开始学习的步数 | 100_000 |
| `policy_frequency` | actor 更新频率 | 1 |
| `gamma` | 折扣因子 | 0.99 |
| `tau` | target 软更新系数 | 0.005 |
| `actor_lr` / `critic_lr` | 学习率 | — |
| `actor_hidden_dim` / `critic_hidden_dim` | 网络隐藏层维度 | — |
| `num_atoms` | Q 分布桶数 | 51 |
| `obs_normalization` | 是否归一化观测 | true |
| `use_layer_norm` | 是否使用 LayerNorm | true |
| `algo_params.alpha_init` | 初始熵系数 | — |
| `algo_params.target_entropy_ratio` | 目标熵比例 | — |

**建议实验**（一次只改一个）：

```bash
uv run python scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco algo.num_envs=512 algo.batch_size=1024
```

---

## 第 7 阶段 · FlashSAC 对比

**目标**：理解 FlashSAC 在 FastSAC 基础上的工程增强。

**核心文件**：

| 文件 | 作用 |
|------|------|
| `conf/offpolicy/algo/flashsac.yaml` | FlashSAC 配置 |
| `src/unilab/algos/torch/flash_sac/learner.py` | FlashSAC Learner |
| `src/unilab/algos/torch/flash_sac/network.py` | FlashSAC 网络模块 |

**FastSAC vs FlashSAC 对比**：

| 对比维度 | FastSAC | FlashSAC |
|----------|---------|----------|
| reward normalizer | 无 | 有 |
| LR scheduler | 无 | 有 |
| 网络结构 | MLP + SiLU + LayerNorm | block 结构 + RMSNorm |
| 探索噪声 | 标准 Gaussian | zeta repeat exploration noise |
| 默认 num_envs | 4096 | 1024 |
| 默认 batch_size | 8192 | 2048 |
| 任务覆盖 | 多任务 | 较少 |
| 多 GPU | 支持 | 当前不支持 |

**关键认知**：FastSAC 是基础主线实现；FlashSAC 是更工程化、专为速度和稳定性优化的版本。

---

## 推荐阅读顺序

按文件依赖关系排列：

```text
1. conf/offpolicy/config.yaml              → 看懂默认配置
2. conf/offpolicy/algo/sac.yaml            → 看懂 SAC 超参数
3. scripts/train_offpolicy.py              → 看懂入口组装
4. src/unilab/algos/torch/fast_sac/learner.py  → 核心算法实现
5. src/unilab/algos/torch/offpolicy/double_buffer_runner.py → 训练循环
6. src/unilab/ipc/replay_buffer.py         → 数据管线
7. src/unilab/algos/torch/flash_sac/       → 对比例外实现
```

---

## 最终小项目（3 个实验）

1. **跑通基线**：`uv run python scripts/train_offpolicy.py algo=sac task=sac/g1_walk_flat/mujoco`
2. **改参数观察**：修改 `batch_size`、`updates_per_step`，观察训练速度和 loss 变化
3. **对比两种 SAC**：跑 `algo=sac` 和 `algo=flashsac` 各一次，写一页笔记说明配置和行为的差异

---

> **核心原则**：先掌握 FastSAC 主线（它是最干净的基础实现），再回头看 FlashSAC 理解了"哪些是 SAC 本身、哪些是工程优化"。

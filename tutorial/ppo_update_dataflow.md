# PPO `update()` 数据流与梯度链详解

> 基于 `rsl_rl/algorithms/ppo.py` 第 228-376 行。

---

## 一、先标清楚：哪些有梯度、哪些是死的

```python
# ===== 来自 batch（全部 detach，不参与梯度） =====
batch.observations          # detach
batch.actions               # detach  ← 关键：动作不变！
batch.old_actions_log_prob  # detach，rollout 时存下的
batch.advantages            # detach，GAE 算好的
batch.returns               # detach，R = A + V_old
batch.values                # detach，rollout 时 V_old(s)
batch.old_distribution_params  # detach

# ===== 当前网络输出（leaf: 有 grad） =====
actions_log_prob  # ← 来自 actor 参数，有 grad
values            # ← 来自 critic 参数，有 grad
entropy           # ← 来自 actor 参数，有 grad
```

**核心原则**：动作 `batch.actions` 始终是 detach 的。PPO 不重新采样动作，只让**当前策略重新给旧动作打分**。打分变了 → 梯度就有了。

critic_obs / obs
returns
old_values

这几个是critic用的

---

## 二、数据流全景图

```text
batch.observations ──┬──→ actor(obs) ──→ 内部分布 (mean, std)
                     │        │
                     │        ├──→ get_log_prob(batch.actions) ──→ actions_log_prob ●
                     │        │                                       │
                     │        └──→ .output_entropy ──────────→ entropy ●
                     │
                     └──→ critic(obs) ──→ values ●

batch.actions ─────────→ get_log_prob(batch.actions) ──→ actions_log_prob ●
                         (只作为 log_prob 的输入，自身 detach)

batch.old_actions_log_prob ──→ ratio = exp(● - ○) ──→ surrogate_loss ●
                                    ↑ grad        ↑ detach

batch.advantages ──→ -A · ratio ──→ surrogate_loss ●
                    ↑ detach, 只当权重

batch.returns ──→ (values ● - returns ○)² ──→ value_loss ●
                  ↑ grad         ↑ detach

batch.values ──→ value_clipped = values ○ + clamp(values ● - values ○)
                 ↑ detach                                    ↑ grad

batch.old_distribution_params ──→ KL(π_old ○ || π_current ●) ──→ 仅调 LR

图例：● = 有梯度    ○ = detach (无梯度)
```

---

## 三、逐条变量的梯度链

### ① `batch.observations` — 最忙的数据

```python
# 第 256-263 行：同时喂给 actor 和 critic
self.actor(batch.observations)   # 设置内部分布
values = self.critic(batch.observations)  # V_current(s)
```

| 路径                                                    | 梯度终点    |
| ------------------------------------------------------- | ----------- |
| `obs → actor → log_prob → ratio → surrogate_loss` | actor 参数  |
| `obs → actor → entropy → loss`                     | actor 参数  |
| `obs → critic → values → value_loss`               | critic 参数 |

一条 obs，两条梯度支流，分别流向 actor 和 critic。

---

### ② `batch.actions` — 只读不写

```text
batch.actions (detach)
     │
     └──→ actor.get_log_prob(batch.actions)
              │
              └── 计算 log π_θ(a|s)，θ 的梯度穿过这里
              │   动作 a 本身不参与梯度
              │
              └──→ actions_log_prob ● ──→ ratio ──→ surrogate_loss
```

动作是**固定的**，但 `log π_θ(a|s)` 对 θ 的导数是：

$$
\frac{\partial \log \pi_\theta(a|s)}{\partial \theta} \neq 0
$$

所以梯度推着 θ 让这个旧动作的 log_prob 朝着 advantage 暗示的方向变。

---

### ③ `batch.old_actions_log_prob` + `batch.advantages` → ratio → surrogate

```python
# 第 297-302 行
ratio = exp(actions_log_prob ● - old_actions_log_prob ○)
#                      ↑ grad          ↑ detach

surrogate         = -advantages ○ * ratio ●
surrogate_clipped = -advantages ○ * clip(ratio ●, 1-ε, 1+ε)
surrogate_loss    = max(surrogate, surrogate_clipped).mean()
```

梯度链（以 advantage > 0，ratio 被 clipped 为例）：

```text
surrogate_loss
  └── surrogate_clipped    ← max 选了这个
       └── -A · clip(ratio, 1-ε, 1+ε)
            └── ratio ● (在 [1-ε, 1+ε] 区间内有梯度)
                 └── exp(actions_log_prob - old_log_prob)
                      └── actions_log_prob ●
                           └── actor 参数
```

| 条件                  | 梯度走哪条                           | 对 actor 的推力          |
| --------------------- | ------------------------------------ | ------------------------ |
| A>0, ratio 在 clip 内 | surrogate                            | 增大 log π              |
| A>0, ratio 超过 1+ε  | surrogate_clipped（clip 截断梯度=0） | **停止**，不再增大 |
| A<0, ratio 在 clip 内 | surrogate                            | 减小 log π              |
| A<0, ratio 低于 1-ε  | surrogate_clipped（clip 截断梯度=0） | **停止**，不再减小 |

**clip 就是梯度断点**：当 ratio 跑出 `[1-ε, 1+ε]`，gradient 归零，actor 不再被这条数据推动。

---

### ④ `batch.returns` + `batch.values` → value_loss

```python
# 第 305-309 行
value_clipped = batch.values ○ + (values ● - batch.values ○).clamp(-ε, ε)
#                 detach          ↑ grad           detach

value_loss = max( (values ● - returns ○)², (value_clipped ● - returns ○)² )
```

梯度链：

```text
value_loss
  └── (values ● - returns ○)²  或  (value_clipped ● - returns ○)²
       └── values ●  ← 梯度只从这里流向 critic 参数
```

**value_loss 的梯度只流向 critic，不影响 actor。** 两条梯度支流完全独立。

---

## 四、梯度链总图

```text
loss.backward()
│
├── surrogate_loss  ──────────────────────────────→ actor 参数
│    └── ratio = π_θ(a|s) / π_old(a|s)
│         └── log π_θ(a|s)  ← 唯一依赖 θ 的项
│
├── + value_coef * value_loss  ──────────────────→ critic 参数
│    └── (V_φ(s) - R_target)²
│         └── V_φ(s)  ← 唯一依赖 φ 的项
│
└── - entropy_coef * entropy  ───────────────────→ actor 参数
     └── H[π_θ(·|s)]
          └── 只依赖 θ
```

```python
# 第 362-376 行：一条 loss.backward() 同时算出两套梯度
loss.backward()  # dL/dθ_actor, dL/dφ_critic 同时存在两个网络的 .grad 里

nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
self.optimizer.step()  # 一个优化器，同时更新 actor 和 critic
```

---

## 五、PPO vs SAC 梯度链对比

|                        | PPO                                      | SAC                                                   |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------- |
| actor 梯度来源         | surrogate (ratio via log_prob) + entropy | Q(s, π(s)) + α·log_prob                            |
| critic 梯度来源        | MSE(V(s), R_target)                      | KL(distribution, Bellman target)                      |
| actor 和 critic 耦合？ | **不耦合**，各自独立               | **耦合**：actor loss 用 Q 值 → 要先更新 critic |
| 优化器                 | 1 个共享                                 | 3 个独立（actor / critic / α）                       |
| loss.backward() 次数   | 每次 mini-batch 1 次                     | update_critic 1 次 + update_actor 1 次                |

---

## 六、一句话总结

> **obs 分两路：一路走 actor 出 log_prob 和 entropy → 构成 surrogate loss；一路走 critic 出 V 值 → 构成 value loss。两条梯度支流完全独立，但通过一次 `loss.backward()` 同时算出，一个优化器同时推两套参数。**

```text
时间轴 ──────────────────────────────────────────────────────→

│←─── rollout 阶段 ───→│←─── update 阶段 ───→│←── rollout ──→│
│                       │                       │
│ actor: eval模式, 冻结  │ actor: train模式, 更新  │
│ critic: eval模式, 冻结 │ critic: train模式, 更新 │
│                       │                       │
│ 采数据 → 存进 storage   │ 读 storage → 算loss → │ 旧 storage 清空
│                       │ backward → optimizer   │ 新策略重新采
│                       │ .step()                │

```

update的第一次循环，policy和critic的参数应该和rollout阶段的是一样的，第一个 mini-batch 时，ratio 就是 1.0

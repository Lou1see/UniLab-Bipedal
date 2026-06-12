# FastSAC 主算法笔记

本文按 UniLab 的 `algo=sac` 主线讲 FastSAC。核心实现入口是：

- `scripts/train_offpolicy.py`
- `src/unilab/algos/torch/fast_sac/learner.py`
- `src/unilab/algos/torch/offpolicy/double_buffer_runner.py`
- `conf/offpolicy/algo/sac.yaml`

FastSAC 和 PPO 的核心差别是：

```text
PPO:
旧 rollout 样本 -> advantage A_i -> pi_new / pi_old -> actor update

FastSAC:
replay buffer 样本 -> critic 学 Q(s, a)
                   -> actor 选让 Q - alpha * log pi 更大的动作
```

---

## 1. Replay batch 里有什么

从 replay buffer 采样一个 batch：

$$
\mathcal{B}
=
\left\{
\left(
s_i,\,
c_i,\,
a_i,\,
r_i,\,
s'_i,\,
c'_i,\,
d_i,\,
t_i
\right)
\right\}_{i=1}^{N}
$$

对应代码：

```python
obs = batch["obs"]                    # s_i, actor obs
critic_obs = batch["critic"]          # c_i, critic obs
actions = batch["actions"]            # a_i, replay 里的旧动作
rewards = batch["rewards"]            # r_i
next_obs = batch["next_obs"]          # s'_i
critic_next_obs = batch["next_critic"] # c'_i
dones = batch["dones"]                # d_i
truncated = batch["truncated"]        # t_i
```

典型 shape：

| 变量 | shape | 意义 |
| --- | ---: | --- |
| `obs` | `[N, obs_dim]` | actor 输入 |
| `critic_obs` | `[N, critic_dim]` | critic 输入 |
| `actions` | `[N, action_dim]` | replay 中真实执行过的动作 |
| `rewards` | `[N]` | reward |
| `next_obs` | `[N, obs_dim]` | 下一步 actor obs |
| `critic_next_obs` | `[N, critic_dim]` | 下一步 critic obs |
| `dones` | `[N]` | terminated 或 truncated |
| `truncated` | `[N]` | time-limit 截断 |

---

## 2. Critic 更新：学 Bellman target

UniLab FastSAC 的 critic 不是输出一个标量 $Q$，而是 distributional Q。每个 critic 输出一组 atom logits：

$$
Z_{\psi}(c_i,a_i)
\in
\mathbb{R}^{K}
$$

代码：

```python
q_outputs = self.qnet(critic_obs, actions)
```

shape 是：

```text
[num_q_networks, N, num_atoms]
```

默认：

```text
num_q_networks = 2
num_atoms = 101
```

也就是 twin Q，每个 Q 网络输出 101 个 atom 的分布 logits。

critic target 先用当前 actor 在下一状态采动作：

$$
a'_i
\sim
\pi_{\theta}(\cdot \mid s'_i)
$$

$$
\log \pi_{\theta}(a'_i \mid s'_i)
$$

代码：

```python
next_actions, next_log_probs, _ = actor.get_actions_and_log_probs(next_obs)
```

SAC 的 soft Bellman target 是：

$$
y_i
=
r_i
+
\gamma b_i
\left(
Q_{\bar{\psi}}(c'_i,a'_i)
-
\alpha \log \pi_{\theta}(a'_i \mid s'_i)
\right)
$$

其中 bootstrap mask 是：

$$
b_i
=
\operatorname{clip}(1-d_i+t_i,0,1)
$$

含义：

```text
真正 terminated: 不 bootstrap
time-limit truncated: 仍然 bootstrap
```

UniLab 代码里把 entropy 项折进 reward：

```python
adjusted_rewards = rewards - gamma * bootstrap * alpha * next_log_probs
target_distributions = qnet_target.projection(
    critic_next_obs,
    next_actions,
    adjusted_rewards,
    bootstrap,
    discount,
)
```

critic loss 是 distributional cross entropy：

$$
L_Q(\psi)
=
-
\mathbb{E}
\sum_k
p^{\mathrm{target}}_{i,k}
\log p_{\psi,k}(c_i,a_i)
$$

代码：

```python
critic_log_probs = F.log_softmax(q_outputs, dim=-1)
critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
qf_loss = critic_losses.mean(dim=1).sum(dim=0)
```

路径图：

```text
replay: c_i, a_i ──→ qnet_psi(c_i, a_i)
                         │
                         └── log_softmax over atoms
                                  │
target_dist from qnet_target ─────┘
                                  │
                             critic loss
                                  │
                              dL / d psi
```

target 分支在 `torch.no_grad()` 里，所以 target actor action 和 target critic distribution 都不反传。

---

## 3. Actor 更新：选高 Q 且别太确定的动作

actor 不用 replay 里的旧动作 `actions` 来算 actor loss。它只用 replay 里的状态 $s_i$，然后当前 actor 重新采动作：

$$
\tilde{a}_i
\sim
\pi_{\theta}(\cdot \mid s_i)
$$

$$
\log \pi_{\theta}(\tilde{a}_i \mid s_i)
$$

代码：

```python
actions, log_probs, log_std = actor.get_actions_and_log_probs(obs)
```

然后把这个新动作丢给当前 critic：

$$
Q_{\psi}(c_i,\tilde{a}_i)
$$

代码：

```python
q_outputs = self.qnet(critic_obs, actions)
q_probs = F.softmax(q_outputs, dim=-1)
q_values = self.qnet.get_value(q_probs)
qf_value = q_values.mean(dim=0)
```

actor loss：

$$
L_{\pi}(\theta)
=
\mathbb{E}_i
\left[
\alpha \log \pi_{\theta}(\tilde{a}_i \mid s_i)
-
Q_{\psi}(c_i,\tilde{a}_i)
\right]
$$

代码：

```python
actor_loss = (alpha.detach() * log_probs - qf_value).mean()
```

路径图：

```text
s_i ──→ actor pi_theta(. | s_i)
            │
            ├── sampled action a_tilde_i ──→ qnet_psi(c_i, a_tilde_i) ──→ Q value
            │                                      │
            └── log pi_theta(a_tilde_i | s_i)      │
                     │                             │
                     └──────── actor loss ◄────────┘
                                  │
                              dL / d theta
```

actor loss 会经过 critic 对 action 的梯度回到 actor，但 actor update 只执行：

```python
actor_optimizer.step()
```

所以 actor loss 不更新 critic 参数。

直觉上：

$$
\min_{\theta}
\left(
\alpha \log \pi_{\theta}
-
Q_{\psi}
\right)
$$

等价于：

$$
\max_{\theta}
\left(
Q_{\psi}
-
\alpha \log \pi_{\theta}
\right)
$$

也就是动作要有高 Q，同时策略保持一定随机性。

---

## 4. Alpha 更新：自动调 entropy 温度

FastSAC 使用：

$$
\alpha
=
\exp(\log \alpha)
$$

代码：

```python
alpha_loss = (-alpha * (next_log_probs + target_entropy)).mean()
```

$\alpha$ 控制探索强度：

| $\alpha$ | 含义 |
| ---: | --- |
| 大 | 更重视 entropy，策略更随机 |
| 小 | 更重视 Q，策略更贪心 |

UniLab 默认配置：

```yaml
algo_params:
  alpha_init: 0.01
  target_entropy_ratio: 0.0
```

---

## 5. Target critic 软更新

critic 更新后，会把当前 critic 慢慢拷贝给 target critic：

$$
\bar{\psi}
\leftarrow
\tau \psi
+
(1-\tau)\bar{\psi}
$$

配置里默认：

```yaml
tau: 0.125
```

target critic 的作用是稳定 Bellman target，不让训练目标本身跟着当前 critic 剧烈抖。

---

## 6. 为什么 FastSAC 不需要 pi_new / pi_old ratio

PPO 需要：

$$
\frac{\pi_{\mathrm{new}}}{\pi_{\mathrm{old}}}
$$

因为 PPO 用旧策略采样的数据直接估计 policy gradient。

SAC 不需要这个 ratio，因为它不是拿旧动作直接做 actor gradient。SAC actor update 用 replay 里的状态 $s_i$，然后当前 actor 重新采：

$$
\tilde{a}_i
\sim
\pi_{\theta}(\cdot \mid s_i)
$$

actor 优化的是当前策略自己的动作：

$$
L_{\pi}
=
\alpha
\log \pi_{\theta}(\tilde{a}_i \mid s_i)
-
Q_{\psi}(c_i,\tilde{a}_i)
$$

replay 里的旧动作 $a_i$ 主要给 critic 学 $Q(c_i,a_i)$ 用。

---

## 7. 最短总结

$$
\boxed{
\text{critic: replay }(s,a,r,s')
\to
\text{soft Bellman target}
\to
Q_{\psi}
}
$$

$$
\boxed{
\text{actor: replay }s
\to
\pi_{\theta}\text{ 重新采动作}
\to
Q_{\psi}(s,\tilde{a})
-
\alpha\log\pi_{\theta}
}
$$

FastSAC 主循环：

```text
collector 不断往 replay 填经验
        │
        ↓
learner 抽 batch 更新 critic
        │
        ↓
隔几步更新 actor 和 alpha
        │
        ↓
软更新 target critic
```

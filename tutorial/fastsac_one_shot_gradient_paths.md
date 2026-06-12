# FastSAC 一镜到底梯度路径

本文模仿 `ppo_one_shot_gradient_paths.md` 的方式，只看 UniLab FastSAC 主算法中每条 loss 的主梯度链。

核心代码位置：

- `src/unilab/algos/torch/fast_sac/learner.py`
- `_critic_loss_tensors()`
- `_actor_loss_tensors()`
- `update_critic()`
- `update_actor()`

UniLab FastSAC 的 critic 是 distributional Q，所以 critic loss 不是普通 MSE，而是 target distribution 与当前 Q distribution 的 cross entropy。

---

## 1. Replay batch 记号

从 replay buffer 采样一个 mini-batch：

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

其中：

| batch 字段 | 数学符号 | shape | 含义 |
| --- | ---: | ---: | --- |
| `batch["obs"]` | $s_i$ | `[N, obs_dim]` | actor 观测 |
| `batch["critic"]` | $c_i$ | `[N, critic_dim]` | critic 观测 |
| `batch["actions"]` | $a_i$ | `[N, action_dim]` | replay 里真实执行过的动作 |
| `batch["rewards"]` | $r_i$ | `[N]` | reward |
| `batch["next_obs"]` | $s'_i$ | `[N, obs_dim]` | 下一步 actor 观测 |
| `batch["next_critic"]` | $c'_i$ | `[N, critic_dim]` | 下一步 critic 观测 |
| `batch["dones"]` | $d_i$ | `[N]` | terminated 或 truncated |
| `batch["truncated"]` | $t_i$ | `[N]` | time-limit truncated |

bootstrap mask：

$$
b_i
=
\operatorname{clip}(1-d_i+t_i,0,1)
$$

含义：

```text
真实终止 terminated: b_i = 0
时间截断 truncated: b_i = 1
普通 transition: b_i = 1
```

---

## 2. Critic 一镜到底

critic 更新用 replay 里的旧动作 $a_i$。它要让当前 critic $Q_{\psi}$ 的分布接近 target critic 给出的 soft Bellman target distribution。

### 2.1 Target 分支：无梯度

target 分支在 `torch.no_grad()` 里：

```python
with torch.no_grad():
    next_actions, next_log_probs, _ = actor.get_actions_and_log_probs(next_obs)
    adjusted_rewards = rewards - gamma * bootstrap * alpha * next_log_probs
    target_distributions = qnet_target.projection(...)
```

当前 actor 在下一状态重新采动作：

$$
a'_i
\sim
\pi_{\theta}(\cdot \mid s'_i)
$$

$$
\log \pi_{\theta}(a'_i \mid s'_i)
$$

soft Bellman target 的标量直觉是：

$$
y_i
=
r_i
+
\gamma b_i
\left(
Q_{\bar{\psi}}(c'_i,a'_i)
-
\alpha
\log \pi_{\theta}(a'_i \mid s'_i)
\right)
$$

但代码中 critic 是 distributional Q，所以实际构造的是 target distribution：

$$
p^{\mathrm{target}}_{i,k}
=
\operatorname{Proj}
\left(
r_i
+
\gamma b_i
\left(
Z_{\bar{\psi}}(c'_i,a'_i)
-
\alpha
\log \pi_{\theta}(a'_i \mid s'_i)
\right)
\right)_k
$$

这里 $p^{\mathrm{target}}_{i,k}$ 不带梯度。

### 2.2 当前 critic 分支：有梯度

当前 critic 输出：

$$
z_{\psi,j}(c_i,a_i)
\in
\mathbb{R}^{K}
$$

其中：

- $j \in \{1,2\}$ 是 twin Q 的编号
- $K$ 是 atom 数，UniLab 默认 `num_atoms = 101`

代码：

```python
q_outputs = self.qnet(critic_obs, actions)
critic_log_probs = F.log_softmax(q_outputs, dim=-1)
```

shape：

```text
q_outputs:        [num_q_networks, N, num_atoms]
critic_log_probs: [num_q_networks, N, num_atoms]
target_dist:      [num_q_networks, N, num_atoms]
```

critic loss：

$$
L_Q(\psi)
=
-
\sum_{j=1}^{2}
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{K}
\underbrace{
p^{\mathrm{target}}_{j,i,k}
}_{\text{target 常数}}
\log
\underbrace{
p_{\psi,j,k}(c_i,a_i)
}_{\text{当前 critic 输出，有梯度}}
$$

其中：

$$
p_{\psi,j,k}(c_i,a_i)
=
\operatorname{softmax}
\left(
z_{\psi,j}(c_i,a_i)
\right)_k
$$

一镜到底梯度：

$$
\nabla_{\psi} L_Q
=
\nabla_{\psi}
\left[
-
\sum_{j=1}^{2}
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{K}
p^{\mathrm{target}}_{j,i,k}
\log
\operatorname{softmax}
\left(
z_{\psi,j}(c_i,a_i)
\right)_k
\right]
$$

传递路径：

```text
L_Q
│
├── target branch, no grad
│    │
│    ├── r_i, d_i, t_i                  batch 常数
│    ├── s'_i ──→ actor pi_theta(. | s'_i)
│    │             └── a'_i, log pi(a'_i | s'_i)
│    └── c'_i, a'_i ──→ target qnet_bar_psi
│                          └── target distribution p_target
│
└── current critic branch, has grad
     │
     ├── c_i                            batch 常数
     ├── a_i                            batch 常数
     └── qnet_psi(c_i, a_i)
             │
             └── logits z_psi
                   │
                   └── log_softmax(z_psi)
                         │
                         └── cross entropy with p_target
                               │
                               └── dL_Q / d psi
```

最短链：

$$
\psi
\to
z_{\psi}(c_i,a_i)
\to
\log\operatorname{softmax}(z_{\psi})
\to
L_Q
\to
\nabla_{\psi}
$$

---

## 3. Actor 一镜到底

actor 更新不使用 replay 里的旧动作 $a_i$。它只使用 replay 里的状态 $s_i$，然后当前 actor 重新采动作：

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

然后当前 critic 评价这个新动作：

$$
Q_{\psi}(c_i,\tilde{a}_i)
$$

UniLab 的 distributional critic 先输出 atom distribution，再取期望：

$$
Q_{\psi,j}(c_i,\tilde{a}_i)
=
\sum_{k=1}^{K}
p_{\psi,j,k}(c_i,\tilde{a}_i)
z_k
$$

twin Q 在代码里取平均：

$$
\bar{Q}_{\psi}(c_i,\tilde{a}_i)
=
\frac{1}{2}
\sum_{j=1}^{2}
Q_{\psi,j}(c_i,\tilde{a}_i)
$$

actor loss：

$$
L_{\pi}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\underbrace{
\alpha
}_{\text{detach}}
\underbrace{
\log \pi_{\theta}(\tilde{a}_i \mid s_i)
}_{\text{当前 actor 输出，有梯度}}
-
\underbrace{
\bar{Q}_{\psi}(c_i,\tilde{a}_i)
}_{\text{critic 对 actor 动作的评价}}
\right]
$$

一镜到底梯度：

$$
\nabla_{\theta} L_{\pi}
=
\nabla_{\theta}
\left[
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\alpha
\log \pi_{\theta}(\tilde{a}_i \mid s_i)
-
\bar{Q}_{\psi}
\left(
c_i,\tilde{a}_i
\right)
\right)
\right]
$$

其中：

$$
\tilde{a}_i
=
f_{\theta}(s_i,\epsilon_i)
$$

这是 reparameterization trick：动作采样写成 actor 参数和噪声的可微函数。

传递路径：

```text
L_pi
│
├── entropy / log_prob term
│    │
│    └── s_i ──→ actor pi_theta(. | s_i)
│                    │
│                    └── log pi_theta(a_tilde_i | s_i)
│                          │
│                          └── alpha * log_prob
│
└── Q term
     │
     ├── c_i                            batch 常数
     └── s_i ──→ actor pi_theta(. | s_i)
                     │
                     └── a_tilde_i
                           │
                           └── qnet_psi(c_i, a_tilde_i)
                                 │
                                 └── Q value
                                       │
                                       └── -Q
```

actor 的两条主链：

$$
\theta
\to
\pi_{\theta}(\cdot \mid s_i)
\to
\log\pi_{\theta}(\tilde{a}_i \mid s_i)
\to
L_{\pi}
$$

$$
\theta
\to
\pi_{\theta}(\cdot \mid s_i)
\to
\tilde{a}_i
\to
Q_{\psi}(c_i,\tilde{a}_i)
\to
L_{\pi}
$$

注意：actor loss 会经过 critic 对 action 的梯度回到 actor，但 `update_actor()` 只执行 `actor_optimizer.step()`，所以这一步不更新 critic 参数。

---

## 4. Alpha 一镜到底

FastSAC 里：

$$
\alpha
=
\exp(\eta)
$$

其中：

$$
\eta
=
\log \alpha
$$

代码里的参数是：

```python
self.log_alpha
```

alpha loss：

$$
L_{\alpha}(\eta)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\exp(\eta)
\left(
\log \pi_{\theta}(a'_i \mid s'_i)
+
\mathcal{H}_{\mathrm{target}}
\right)
$$

代码：

```python
alpha_loss = (-self.log_alpha.exp() * (next_log_probs + self.target_entropy)).mean()
```

这里 `next_log_probs` 来自 critic target 分支，并且在 `update_critic()` 里是 detach 后返回的：

```python
return qf_loss, target_q_max, target_q_min, next_log_probs.detach()
```

所以 alpha 更新只更新 $\eta=\log\alpha$。

传递路径：

```text
L_alpha
│
├── next_log_probs                  detach 常数
├── target_entropy                  常数
└── log_alpha eta
        │
        └── alpha = exp(eta)
              │
              └── -alpha * (next_log_probs + target_entropy)
                    │
                    └── dL_alpha / d eta
```

最短链：

$$
\eta
\to
\alpha=\exp(\eta)
\to
L_{\alpha}
\to
\nabla_{\eta}
$$

---

## 5. 总图

```text
critic update
-------------
batch c_i, a_i ──→ qnet_psi(c_i, a_i) ──→ log_softmax ──┐
                                                         ├── L_Q ──→ d psi
target branch no_grad ──→ target distribution ──────────┘


actor update
------------
batch s_i ──→ actor pi_theta(. | s_i)
                 │
                 ├── log pi_theta(a_tilde_i | s_i) ─────┐
                 │                                       ├── L_pi ──→ d theta
                 └── a_tilde_i ──→ qnet_psi(c_i,a_tilde) ┘


alpha update
------------
log_alpha eta ──→ alpha = exp(eta)
                       │
next_log_probs detach ─┤
target_entropy const ──┘
                       │
                    L_alpha ──→ d eta
```

---

## 6. 最短总结

$$
\boxed{
\text{critic: }
(c_i,a_i,r_i,s'_i,c'_i,d_i,t_i)
\to
p^{\mathrm{target}}
\quad\text{vs}\quad
p_{\psi}(c_i,a_i)
\to
L_Q
\to
\nabla_{\psi}
}
$$

$$
\boxed{
\text{actor: }
s_i
\to
\tilde{a}_i \sim \pi_{\theta}
\to
\alpha\log\pi_{\theta}(\tilde{a}_i|s_i)
-
Q_{\psi}(c_i,\tilde{a}_i)
\to
L_{\pi}
\to
\nabla_{\theta}
}
$$

$$
\boxed{
\text{alpha: }
\log\alpha
\to
\alpha
\to
L_{\alpha}
\to
\nabla_{\log\alpha}
}
$$

---

## SAC Actor Loss 中 `α · log π` 的物理意义

### 完整 Actor Loss

$$
L_{\text{actor}} = \mathbb{E}_{a \sim \pi}\Big[\alpha \cdot \log \pi(a|s) - Q(s, a)\Big]
$$

**最小化**这个 loss，两项角力：

| 项 | 方向 | 效果 |
|----|------|------|
| `α · log π` | log π < 0，α > 0 → 此项为负 → 让 log π 更负 → **降低确定性** | 熵阻力 |
| `- Q(s, a)` | 让 `-Q` 小 → 推高 π 在高 Q 动作上的概率 → **往高分区聚** | Q 引力 |

---

### 从 KL 散度推导：策略蒸馏

SAC 的 actor 想让自己逼近一个"以 Q 为能量的 Boltzmann 分布"：

$$
p^*(a|s) = \frac{\exp(Q(s, a)/\alpha)}{Z(s)}, \quad Z(s) = \sum_a \exp(Q/\alpha)
$$

用 KL 散度衡量距离：

$$
\begin{aligned}
D_{\text{KL}}(\pi \parallel p^*) &= \mathbb{E}_{a\sim\pi}\big[\log\pi(a) - \log p^*(a)\big] \\
&= \mathbb{E}_{a\sim\pi}\left[\log\pi(a) - \frac{Q(s,a)}{\alpha} + \log Z(s)\right]
\end{aligned}
$$

`log Z(s)` 是常数，扔掉，乘 α 即得 actor loss：

$$
\min_\pi D_{\text{KL}} \iff \min_\pi \mathbb{E}\big[\alpha \cdot \log\pi(a) - Q(s,a)\big]
$$

**最小化 `α·log π - Q` = 让策略尽量贴近 `exp(Q/α)` 分布。**Q 值高的动作概率大，Q 值低的动作概率小。

---

### 从梯度视角理解

用 rsample（a = μ + σ·ε）对 θ 求导：

$$
\frac{\partial L}{\partial \theta} = \alpha \cdot \frac{\partial \log\pi}{\partial\theta} - \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial\theta}
$$

- **Q 引导项**：把动作往 Q 值高的方向推
- **熵正则项**：阻力，防止概率全堆在一点

平衡点：

$$
\alpha \cdot \frac{\partial \log\pi}{\partial\theta} = \frac{\partial Q}{\partial a} \cdot \frac{\partial a}{\partial\theta}
$$

---

### 一句话类比

| 项 | 类比 | 作用 |
|----|------|------|
| `-Q(s, a)` | **引力** | 往高分区走 |
| `α · log π` | **熵阻力** | 别把所有鸡蛋放一个篮子里 |
| α | **阻力系数** | α 大 → 更分散、更探索；α 小 → 更贪心、更确定 |

> **Q 项告诉策略"哪儿好"，`α·log π` 告诉策略"别太信"。两者拉锯的平衡点，就是 SAC 学到的策略。**

---

## Q 引力与熵阻力的动态平衡

### 拉锯过程
随着算法的推演  具有最大值的Q(a,s)，所对应的策略pi(a,s)将会被调高。但是由于熵阻力项的存在，即pi(a,s)一旦变大，loss又会变大，优化器又想去压loss，间接导致了，pi(a,s)又会被压下去，达到一个动态平衡。

随着算法推演，具有最大 Q 值的动作所对应的 π(a|s) 会被 Q 引力调高。但由于熵阻力项 `α·log π` 的存在：

- π 变大 → log π 从负值靠近 0 → `α·log π` 变大 → **loss 变大**
- 优化器又想去压 loss → **π 又被压回来**

两个力同时作用，达到动态平衡。

### 力的分布

```
                    Q 引力                         熵阻力
                      →                              ←
π(a|s) ≈ 0  │   Q 很大，拼命往上拽          log π → -∞，阻力无穷大
            │
π(a|s) ≈ 0.3│   Q 还在拽，但劲变小了          log π ≈ -1.2，阻力适中
            │
π(a|s) ≈ 0.8│   Q 引力很弱了                  log π ≈ -0.22，阻力很小
            │
π(a|s) → 1  │   几乎没引力                    阻力 → 0
```

均衡点不是 π=1，而是两个力的交汇处。

### 均衡条件

对策略参数 θ 的梯度：

$$
\frac{\partial L}{\partial\theta} = \frac{\partial}{\partial\theta}\mathbb{E}_{a\sim\pi}\big[\alpha\log\pi - Q\big]
$$

梯度为零时停止更新：

$$
\alpha \cdot \frac{\partial}{\partial\theta}\mathbb{E}\big[\log\pi\big] = \frac{\partial}{\partial\theta}\mathbb{E}\big[Q\big]
$$

**熵梯度和 Q 的梯度正好对消，参数不动了。**

### 均衡策略

可证明梯度为零时的最优策略为：

$$
\pi^*(a|s) = \frac{\exp(Q(s,a)/\alpha)}{Z}
$$

| α 值 | 均衡策略 |
|------|---------|
| α → 0（没熵阻力） | 全部概率堆在最高 Q 动作 → 贪心策略 |
| α 适中 | 概率按 `exp(Q/α)` 分布，Q 高的概率大 |
| α → ∞（超强探索） | 接近均匀分布，不管 Q 了 |

### 为什么 π 不会塌缩到 0

关键在于 `E_{a~π}[α·log π] = -α·H[π]`。期望下的 log π 恰好是负熵。

想把某个动作的概率压到 0，你必须让分布变窄（低熵）。但 `-α·H` 在惩罚低熵：

| 分布状态 | 熵 H | `-α·H` | 效果 |
|---------|------|--------|------|
| 坍塌（全押一个动作） | → 0 | → 0 | 没有熵补贴了，loss 变高 |
| 均匀探索 | 大 | 很大的负值 | loss 很低 |

**压概率 → 低熵 → 熵补贴消失 → loss 上升 → 停止压缩。** 这就是负反馈循环。

### α 的自动调节也在帮忙

α 不是固定的，通过 alpha loss 动态调整：

```
策略太确定了 → log π 不够负 → alpha_loss 上升 → α 变大 → 熵阻力变强 → 策略被拽回探索
策略太随机了 → log π 太负 → alpha_loss 下降 → α 变小 → 熵阻力变弱 → 策略更贪心
```

> **Q 把策略往高分区拽，熵项把它往回拉。两个力同时作用，在 `π ∝ exp(Q/α)` 处停住——Q 越高概率越大，但不会独吞，α 决定了"分享"的程度。**
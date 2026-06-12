# PPO 一镜到底梯度路径

本文只看 PPO `update()` 里的主梯度链，并且忽略所有限位项：

- 忽略 policy ratio clipping
- 忽略 value clipping

目标是把 actor 和 critic 各自从最终损失一路追到 batch 固定变量与当前网络新计算变量。

---

## 1. Actor 一镜到底

actor 分支用到的 batch 固定量是：

$$
\left(
s_i,\,
a_i,\,
\log \pi_{\mathrm{old}}(a_i \mid s_i),\,
A_i
\right)
\in
\mathcal{B}
$$

其中：

| batch 字段 | 数学符号 | 含义 |
| --- | ---: | --- |
| `batch.observations` | $s_i$ | rollout 阶段采到的观测 |
| `batch.actions` | $a_i$ | rollout 阶段旧 policy 采样出的动作 |
| `batch.old_actions_log_prob` | $\log \pi_{\mathrm{old}}(a_i \mid s_i)$ | 旧 policy 对旧动作的 log probability |
| `batch.advantages` | $A_i$ | GAE 算出的 advantage |

update 阶段 actor 当前重新计算的量是：

$$
\log \pi_{\theta}(a_i \mid s_i),
\qquad
\pi_{\theta}(\cdot \mid s_i),
\qquad
\mathcal{H}\left[\pi_{\theta}(\cdot \mid s_i)\right]
$$

忽略 clip 后，actor loss 一条公式写成：

$$
L_{\mathrm{actor}}(\theta)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
A_i
\exp
\left(
\log \pi_{\theta}(a_i \mid s_i)
-
\log \pi_{\mathrm{old}}(a_i \mid s_i)
\right)
-
c_H
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{H}
\left[
\pi_{\theta}(\cdot \mid s_i)
\right]
$$

把 batch 常数和当前网络输出都标进去：

$$
\nabla_{\theta} L_{\mathrm{actor}}
=
\nabla_{\theta}
\left[
-
\frac{1}{N}
\sum_{i=1}^{N}
\underbrace{A_i}_{\text{batch 常数}}
\exp
\left(
\underbrace{\log \pi_{\theta}(a_i \mid s_i)}_{\text{当前 actor 输出，有梯度}}
-
\underbrace{\log \pi_{\mathrm{old}}(a_i \mid s_i)}_{\text{batch 常数}}
\right)
-
c_H
\frac{1}{N}
\sum_{i=1}^{N}
\underbrace{
\mathcal{H}
\left[
\pi_{\theta}(\cdot \mid s_i)
\right]
}_{\text{当前 actor 输出，有梯度}}
\right]
$$

传递路径：

```text
L_actor
│
├── surrogate term
│    │
│    ├── A_i                           batch 常数
│    ├── log pi_old(a_i | s_i)         batch 常数
│    ├── a_i                           batch 常数
│    └── s_i ──→ actor pi_theta(. | s_i)
│                    │
│                    └── log pi_theta(a_i | s_i)
│                          │
│                          └── exp(log pi_theta - log pi_old)
│                                │
│                                └── -A_i * ratio
│
└── entropy term
     │
     └── s_i ──→ actor pi_theta(. | s_i)
                     │
                     └── H[pi_theta(. | s_i)]
```

actor 的两条梯度主链是：

$$
\theta
\to
\pi_{\theta}(\cdot \mid s_i)
\to
\log \pi_{\theta}(a_i \mid s_i)
\to
L_{\mathrm{actor}}
$$

$$
\theta
\to
\pi_{\theta}(\cdot \mid s_i)
\to
\mathcal{H}\left[\pi_{\theta}(\cdot \mid s_i)\right]
\to
L_{\mathrm{actor}}
$$

---

## 2. Critic 一镜到底

critic 分支用到的 batch 固定量是：

$$
\left(
s_i,\,
R_i
\right)
\in
\mathcal{B}
$$

其中：

| batch 字段 | 数学符号 | 含义 |
| --- | ---: | --- |
| `batch.observations` / `critic_obs` | $s_i$ | critic 输入观测 |
| `batch.returns` | $R_i$ | critic 的监督目标 |

critic 当前重新计算的量是：

$$
V_{\phi}(s_i)
$$

忽略 value clipping 后，critic loss 一条公式写成：

$$
L_{\mathrm{critic}}(\phi)
=
c_V
\frac{1}{N}
\sum_{i=1}^{N}
\left(
V_{\phi}(s_i)
-
R_i
\right)^2
$$

把 batch 常数和当前网络输出都标进去：

$$
\nabla_{\phi} L_{\mathrm{critic}}
=
\nabla_{\phi}
\left[
c_V
\frac{1}{N}
\sum_{i=1}^{N}
\left(
\underbrace{V_{\phi}(s_i)}_{\text{当前 critic 输出，有梯度}}
-
\underbrace{R_i}_{\text{batch target，常数}}
\right)^2
\right]
$$

传递路径：

```text
L_critic
│
└── value term
     │
     ├── R_i                           batch 常数
     └── s_i ──→ critic V_phi(s_i)
                     │
                     └── (V_phi(s_i) - R_i)^2
```

critic 的梯度主链是：

$$
\phi
\to
V_{\phi}(s_i)
\to
\left(
V_{\phi}(s_i)-R_i
\right)^2
\to
L_{\mathrm{critic}}
$$

---

## 3. 最短总结

$$
\boxed{
\text{actor: }
\left(s_i,a_i,\log\pi_{\mathrm{old}},A_i\right)
+
\log\pi_{\theta}(a_i \mid s_i)
+
\mathcal{H}[\pi_{\theta}]
\to
L_{\mathrm{actor}}
\to
\nabla_{\theta}
}
$$

$$
\boxed{
\text{critic: }
\left(s_i,R_i\right)
+
V_{\phi}(s_i)
\to
L_{\mathrm{critic}}
\to
\nabla_{\phi}
}
$$

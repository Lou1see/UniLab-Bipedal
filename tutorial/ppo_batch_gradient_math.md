# PPO batch 变量与梯度传播公式图

这篇笔记只看 PPO `update()` 阶段：mini-batch 里哪些量是固定样本 / 固定目标，哪些量由当前 actor / critic 重新 forward 得到，并参与梯度传播。

---

## 1. Mini-batch 记号

把一个 mini-batch 记成：

$$
\mathcal{B}
=
\left\{
\left(
s_i,\,
a_i,\,
\log \pi_{\mathrm{old}}(a_i \mid s_i),\,
A_i,\,
R_i,\,
V_{\mathrm{old},i}
\right)
\right\}_{i=1}^{N}
$$

其中：

| batch 字段                        |                                  数学符号 | 是否带梯度 | 含义                                 |
| --------------------------------- | ----------------------------------------: | ---------: | ------------------------------------ |
| `batch.observations`            |                                   $s_i$ |         否 | rollout 阶段采到的状态 / 观测        |
| `batch.actions`                 |                                   $a_i$ |         否 | rollout 阶段旧 policy 采样出的动作   |
| `batch.old_actions_log_prob`    | $\log \pi_{\mathrm{old}}(a_i \mid s_i)$ |         否 | 旧 policy 对旧动作的 log probability |
| `batch.advantages`              |                                   $A_i$ |         否 | GAE 算出的 advantage                 |
| `batch.returns`                 |                                   $R_i$ |         否 | critic 的监督目标                    |
| `batch.values` / `old_values` |                    $V_{\mathrm{old},i}$ |         否 | rollout 阶段旧 critic 的估值         |

update 阶段当前网络重新计算：

$$
\log \pi_{\theta}(a_i \mid s_i)
$$

$$
V_{\phi}(s_i)
$$

$$
\mathcal{H}\left[\pi_{\theta}(\cdot \mid s_i)\right]
$$

这里 $\theta$ 是 actor 参数，$\phi$ 是 critic 参数。batch 里的变量都是常数；真正产生梯度的是当前网络输出。

---

## 2. 总图

```text
batch: s_i, a_i, log pi_old, A_i, R_i, V_old
          │
          ├──────────────→ actor pi_theta(. | s_i)
          │                    │
          │                    ├── log pi_theta(a_i | s_i)
          │                    │        │
          │                    │        └── surrogate loss ─────→ actor 参数 theta
          │                    │
          │                    └── entropy H[pi_theta(. | s_i)]
          │                             │
          │                             └── entropy loss ───────→ actor 参数 theta
          │
          └──────────────→ critic V_phi(s_i)
                               │
                               └── value loss ─────────────────→ critic 参数 phi
```

---

## 3. Actor: Surrogate Loss

PPO 先计算当前 policy 与旧 policy 的概率比：

$$
r_i(\theta)
=
\frac{\pi_{\theta}(a_i \mid s_i)}
{\pi_{\mathrm{old}}(a_i \mid s_i)}
=
\exp\left(
\log \pi_{\theta}(a_i \mid s_i)
-
\log \pi_{\mathrm{old}}(a_i \mid s_i)
\right)
$$

clipped surrogate loss 是：

$$
L_{\mathrm{sur}}(\theta)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\min
\left(
r_i(\theta) A_i,\,
\operatorname{clip}
\left(
r_i(\theta), 1-\epsilon, 1+\epsilon
\right)
A_i
\right)
$$

梯度是：

$$
\nabla_{\theta} L_{\mathrm{sur}}
$$

传播链：

```text
theta
  │
  └── actor pi_theta(. | s_i)
        │
        └── log pi_theta(a_i | s_i)
              │
              └── r_i(theta)
                    │
                    └── L_sur(theta)
```

不参与梯度的 batch 常数：

$$
s_i,\quad
a_i,\quad
\log \pi_{\mathrm{old}}(a_i \mid s_i),\quad
A_i
$$

直觉：

$$
A_i > 0
\Rightarrow
\text{提高 } \pi_{\theta}(a_i \mid s_i)
$$

$$
A_i < 0
\Rightarrow
\text{降低 } \pi_{\theta}(a_i \mid s_i)
$$

---

## 4. Critic: Value Loss

不开 value clipping 时：

$$
L_V(\phi)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left(
V_{\phi}(s_i) - R_i
\right)^2
$$

梯度是：

$$
\nabla_{\phi} L_V
$$

传播链：

```text
phi
  │
  └── critic V_phi(s_i)
        │
        └── L_V(phi)
```

不参与梯度的 batch 常数：

$$
s_i,\quad
R_i
$$

开启 clipped value loss 时，先计算：

$$
V_i^{\mathrm{clip}}
=
V_{\mathrm{old},i}
+
\operatorname{clip}
\left(
V_{\phi}(s_i)-V_{\mathrm{old},i},
-\epsilon,
+\epsilon
\right)
$$

然后：

$$
L_V(\phi)
=
\frac{1}{N}
\sum_{i=1}^{N}
\max
\left(
\left(V_{\phi}(s_i)-R_i\right)^2,\,
\left(V_i^{\mathrm{clip}}-R_i\right)^2
\right)
$$

这里 $V_{\mathrm{old},i}$ 是 batch 里存下来的旧 critic value，不带梯度；只有 $V_{\phi}(s_i)$ 带梯度。

---

## 5. Actor: Entropy Loss

entropy 项鼓励 policy 分布保持一定随机性：

$$
L_H(\theta)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{H}
\left[
\pi_{\theta}(\cdot \mid s_i)
\right]
$$

梯度是：

$$
\nabla_{\theta} L_H
$$

传播链：

```text
theta
  │
  └── actor pi_theta(. | s_i)
        │
        └── H[pi_theta(. | s_i)]
              │
              └── L_H(theta)
```

entropy 不需要 $a_i$，因为它看的是当前 policy 的整个动作分布，而不是某个 rollout 动作的概率。

---

## 6. 总 Loss 与梯度分裂

代码里的总 loss 可以写成：

$$
L(\theta,\phi)
=
L_{\mathrm{sur}}(\theta)
+
c_V L_V(\phi)
-
c_H
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{H}
\left[
\pi_{\theta}(\cdot \mid s_i)
\right]
$$

其中 $c_V$ 对应 `value_loss_coef`，$c_H$ 对应 `entropy_coef`。

梯度分裂成两路：

$$
\nabla_{\theta} L
=
\nabla_{\theta} L_{\mathrm{sur}}
+
\nabla_{\theta} L_H
$$

$$
\nabla_{\phi} L
=
c_V \nabla_{\phi} L_V
$$

最终图：

```text
                         ┌── log pi_theta(a_i | s_i)
s_i ──→ actor pi_theta ──┤
                         └── H[pi_theta(. | s_i)]
          │                    │
a_i ──────┘                    │
                               │
log pi_old ───→ ratio ─────────┤
A_i ──────────→ surrogate ─────┴────→ dL / d theta

s_i ──→ critic V_phi(s_i)
          │
R_i ──────┼──→ value loss ──────────→ dL / d phi
V_old ────┘      only if value clip
```

一句话总结：

> batch 给的是固定样本和固定目标；actor 的梯度来自当前 $\log \pi_{\theta}(a_i \mid s_i)$ 和 entropy，critic 的梯度来自当前 $V_{\phi}(s_i)$。

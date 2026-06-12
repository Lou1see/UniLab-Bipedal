# PPO 梯度传播：从 log_prob 到 MLP 权重

> 基于 `rsl_rl/algorithms/ppo.py` 第 262 行、`rsl_rl/models/mlp_model.py`、`rsl_rl/modules/distribution.py`

---

## 调用链

```python
# ppo.py 第 262 行
actions_log_prob = self.actor.get_output_log_prob(batch.actions)
#                              ↓
# mlp_model.py 第 153-155 行
def get_output_log_prob(self, outputs):
    return self.distribution.log_prob(outputs)
#                              ↓
# distribution.py 第 215-217 行
def log_prob(self, outputs):
    return self._distribution.log_prob(outputs).sum(dim=-1)
#               ↓
# torch.distributions.Normal.log_prob()
```

---

## `Normal.log_prob` 的公式

```python
# torch/distributions/normal.py
def log_prob(self, value):
    var = self.scale ** 2
    log_scale = self.scale.log()
    return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
```

数学形式：

$$
\log \pi(a | s) = -\frac{(a - \mu)^2}{2\sigma^2} - \log\sigma - \log\sqrt{2\pi}
$$

---

## 梯度流向

三个变量，两个有 grad：

| 变量 | 来自 | 有 grad？ |
|------|------|----------|
| `a` (batch.actions) | rollout 时存下的旧动作，已 detach | ❌ 无 |
| `μ` (mean) | **MLP 的输出** | ✅ 有 |
| `σ` (std) | **nn.Parameter**（或 MLP 输出） | ✅ 有 |

对 **μ** 求导：

$$
\frac{\partial \log\pi}{\partial \mu} = \frac{a - \mu}{\sigma^2}
$$

**直觉**：如果动作 a > 均值 μ，梯度为正 → 增大 μ 向 a 靠近；反之减小 μ。

对 **σ** 求导：

$$
\frac{\partial \log\pi}{\partial \sigma} = \frac{(a - \mu)^2}{\sigma^3} - \frac{1}{\sigma}
$$

---

## mean 到 MLP 的传递

`mean` 是恒等映射，梯度无损穿过：

```python
# GaussianDistribution.update()
def update(self, mlp_output):
    mean = mlp_output         # ← 恒等！d(mean)/d(mlp_output) = 1
    std = self.std_param.expand_as(mean)
    self._distribution = Normal(mean, std)
```

梯度链：

```
$dL/d\mu \to d\mu/d\text{mlp\_output} = 1 \to dL/d\text{mlp\_output} \to$ MLP 权重
```

---

## 完整梯度链

```
loss
 │
 ├── surrogate_loss = -A · exp(log π - log π_old)
 │    │
 │    └── actions_log_prob
 │         │
 │         └── Normal.log_prob(action, mean, std)
 │              │
 │              ├── d/d(mean) = (action - mean) / σ²
 │              │    └── MLP 权重（通过恒等 mean=mlp_output 直接反传）
 │              │
 │              └── d/d(std) = (action - mean)²/σ³ - 1/σ
 │                   └── nn.Parameter(std) 直接更新
 │
 └── - entropy_coef * entropy
      │
     └── entropy = 0.5 · log(2πe·σ²)
          └── d(entropy)/dσ = 1/σ → 只流向 std
```

---

## 核心要点

> **动作是死的（detach），分布是活的（有 grad）。log_prob 对 mean/std 求导，梯度沿 MLP 反传。一条 `get_output_log_prob` 同时算出 actor 的 surrogate 梯度 + entropy 梯度，通过一次 `loss.backward()` 更新 MLP 所有权重。**

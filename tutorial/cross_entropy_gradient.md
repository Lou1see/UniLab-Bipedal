# 交叉熵梯度推导：为什么 `p_current` 会逼近 `p_target`

> 解答：FastSAC 的 critic loss 用交叉熵，梯度 = $p_{\text{current}} - p_{\text{target}}$，直接推着当前分布向目标分布靠近。

---

## 给定数据

$$
p_{\text{target}} = [0.1, 0.5, 0.4], \quad
p_{\text{current}} = \text{softmax}(z) = [0.3, 0.3, 0.4]
$$

logits：$z = [\ln 0.3, \ln 0.3, \ln 0.4] = [-1.204, -1.204, -0.916]$

---

## Step 1: 交叉熵 Loss 计算

$$
\begin{aligned}
L &= -\sum_{k=1}^{3} p_t(k) \cdot \ln p_c(k) \\[4pt]
  &= -[0.1 \cdot \ln 0.3 + 0.5 \cdot \ln 0.3 + 0.4 \cdot \ln 0.4] \\[4pt]
  &= -[0.1 \cdot (-1.204) + 0.5 \cdot (-1.204) + 0.4 \cdot (-0.916)] \\[4pt]
  &= -[-0.120 - 0.602 - 0.366] = 1.088
\end{aligned}
$$

---

## Step 2: softmax 的导数

$$
p_c(k) = \frac{e^{z_k}}{\sum_i e^{z_i}}, \quad 
\frac{\partial p_c(k)}{\partial z_j} = p_c(k) \cdot (\delta_{kj} - p_c(j))
$$

其中 $\delta_{kj} = 1$ if $k = j$, else $0$。

---

## Step 3: 交叉熵对 logit 求导

$$
\begin{aligned}
\frac{\partial L}{\partial z_j} 
&= -\sum_k p_t(k) \cdot \frac{\partial \ln p_c(k)}{\partial z_j} \\[4pt]
&= -\sum_k p_t(k) \cdot \frac{1}{p_c(k)} \cdot \frac{\partial p_c(k)}{\partial z_j} \\[4pt]
&= -\sum_k p_t(k) \cdot \frac{1}{p_c(k)} \cdot p_c(k) \cdot (\delta_{kj} - p_c(j)) \\[4pt]
&= -\sum_k p_t(k) \cdot (\delta_{kj} - p_c(j)) \\[4pt]
&= -\left[ p_t(j) \cdot 1 - p_c(j) \cdot \sum_k p_t(k) \right] \\[4pt]
&= -\left[ p_t(j) - p_c(j) \cdot 1 \right] \\[4pt]
&= p_c(j) - p_t(j)
\end{aligned}
$$

---

## Step 4: 数值代入

$$
\begin{aligned}
\frac{\partial L}{\partial z_1} &= 0.3 - 0.1 = +0.20 \quad \leftarrow \text{太多，压低} \\[4pt]
\frac{\partial L}{\partial z_2} &= 0.3 - 0.5 = -0.20 \quad \leftarrow \text{太少，推高} \\[4pt]
\frac{\partial L}{\partial z_3} &= 0.4 - 0.4 = \ \ 0.00 \quad \leftarrow \text{刚好，不动}
\end{aligned}
$$

---

## Step 5: 一步梯度下降的效果

$lr = 0.5$：

$$
z^{\text{new}} = z - lr \cdot \frac{\partial L}{\partial z}
$$

$$
\begin{aligned}
z_1^{\text{new}} &= -1.204 - 0.5 \cdot (+0.20) = -1.304 \quad \leftarrow \text{变小} \\[4pt]
z_2^{\text{new}} &= -1.204 - 0.5 \cdot (-0.20) = -1.104 \quad \leftarrow \text{变大} \\[4pt]
z_3^{\text{new}} &= -0.916 - 0.5 \cdot 0 = -0.916 \quad\quad\;\; \leftarrow \text{不变}
\end{aligned}
$$

更新后重新 softmax：

$$
p_c^{\text{new}} \approx [0.23, 0.35, 0.42]
$$

与原分布 $[0.3, 0.3, 0.4]$ 相比：

- **$p_c(1)$：0.30 → 0.23**，向 target 0.1 靠近 ✅
- **$p_c(2)$：0.30 → 0.35**，向 target 0.5 靠近 ✅
- **$p_c(3)$：0.40 → 0.42**，基本不动（本来就和 target 0.4 一致）

---

## 物理直觉

```
target  = [0.1,  0.5,  0.4]     ← 标答
current = [0.3,  0.3,  0.4]     ← 现状
梯度     = [+0.2, -0.2, 0.0]    ← p_c - p_t
          └太多┘ └太少┘ └刚好┘

lr=0.5 后退一步:
current'≈ [0.23, 0.35, 0.42]   ← 更接近 target 了 ✓
```

---

## 在 FastSAC 中的位置

```python
# learner.py 第 616-618 行
critic_log_probs = F.log_softmax(q_outputs, dim=-1)   # ln(p_current)
critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
#               ↑ 交叉熵

# loss.backward() 后每个 logit 的梯度 = p_current - p_target
# optimizer.step() → p_current 向 p_target 靠近
```

> **梯度 = $p_c - p_t$，既管方向又管大小。交叉熵+softmax 的组合自带"差多少推多少"的纠正信号。高的压低、低的推高、刚好的不动。每轮 SGD 都在缩减偏差，直到 $p_c = p_t$ 时梯度全零收敛。**

---

## 附录：K=2 时从 Loss 到 z 的完整链式求导

### 定义

$$
z = [z_1, z_2], \quad
s_k = \frac{e^{z_k}}{e^{z_1} + e^{z_2}}, \quad
L = -\big[p_t(1) \cdot \ln s_1 + p_t(2) \cdot \ln s_2\big]
$$

### softmax 偏导数

$$
\frac{\partial s_1}{\partial z_1} = s_1(1 - s_1), \quad
\frac{\partial s_1}{\partial z_2} = -s_1 s_2
$$

$$
\frac{\partial s_2}{\partial z_1} = -s_1 s_2, \quad
\frac{\partial s_2}{\partial z_2} = s_2(1 - s_2)
$$

### 链式求导 ∂L/∂z₁

L 是两项的和，z₁ 同时出现在 s₁ 和 s₂ 中：

$$
\frac{\partial L}{\partial z_1} =
\underbrace{\frac{\partial L}{\partial \ln s_1} \cdot \frac{\partial \ln s_1}{\partial z_1}}_{\text{第一项 }} +
\underbrace{\frac{\partial L}{\partial \ln s_2} \cdot \frac{\partial \ln s_2}{\partial z_1}}_{\text{第二项}}
$$

逐项展开：

第一项：

$$
\begin{aligned}
\frac{\partial L}{\partial \ln s_1} &= -p_t(1) \\[4pt]
\frac{\partial \ln s_1}{\partial z_1} &= \frac{1}{s_1} \cdot \frac{\partial s_1}{\partial z_1}
= \frac{1}{s_1} \cdot s_1(1 - s_1) = 1 - s_1
\end{aligned}
$$

第一项 $= (-p_t(1)) \cdot (1 - s_1) = -p_t(1) + p_t(1) \cdot s_1$

第二项：

$$
\begin{aligned}
\frac{\partial L}{\partial \ln s_2} &= -p_t(2) \\[4pt]
\frac{\partial \ln s_2}{\partial z_1} &= \frac{1}{s_2} \cdot \frac{\partial s_2}{\partial z_1}
= \frac{1}{s_2} \cdot (-s_1 s_2) = -s_1
\end{aligned}
$$

第二项 $= (-p_t(2)) \cdot (-s_1) = p_t(2) \cdot s_1$

### 合并

$$
\begin{aligned}
\frac{\partial L}{\partial z_1}
&= \big(-p_t(1) + p_t(1) \cdot s_1\big) + \big(p_t(2) \cdot s_1\big) \\[4pt]
&= -p_t(1) + s_1 \cdot \underbrace{(p_t(1) + p_t(2))}_{=1} \\[4pt]
&= -p_t(1) + s_1 \\[4pt]
&= s_1 - p_t(1)
\end{aligned}
$$

同理：

$$
\frac{\partial L}{\partial z_2} = s_2 - p_t(2)
$$

### 通用形式

对任意 K 个原子：

$$
\boxed{\frac{\partial L}{\partial z_j} = s_j - p_t(j) = p_c(j) - p_t(j)}
$$

### 数值验证

```
s = p_c = [0.3, 0.7],  p_t = [0.1, 0.9]

∂L/∂z₁ = 0.3 - 0.1 = +0.20  ← 太多了，压低 z₁
∂L/∂z₂ = 0.7 - 0.9 = -0.20  ← 太少了，推高 z₂
```

验证：$+0.20 + (-0.20) = 0$，梯度之和为零 → 概率单纯形约束自动满足。

### 关键：求和消掉了交叉项

链式法则穿过 softmax 时，z₁ 会接收到来自**所有** s_k 的梯度（因为 softmax 分母是全局的）。但加总后交叉项恰好消掉，只剩下本地的 $s_j - p_t(j)$。

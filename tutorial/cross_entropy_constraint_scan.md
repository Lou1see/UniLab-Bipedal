# 交叉熵几何图：沿概率约束线扫描 current distribution

> 这张图用二分类例子解释：$p_{\text{current}}$ 是概率向量，必须落在 $x+y=1$ 上；而 $-\log p_{\text{current}}$ 是 surprise 向量，不再受概率约束。交叉熵就是 $p_{\text{target}}$ 和这个 surprise 向量的点积。

---

## 图

![Cross entropy constraint scan](imagefastsac/cross_entropy_2class_constraint_scan.png)

---

## 左图怎么看

固定目标分布：

$$
p_{\text{target}} = [0.2,\ 0.8]
$$

二分类 current distribution 写成：

$$
p_{\text{current}} = [x,\ y]
$$

因为它是概率分布，所以必须满足：

$$
x+y=1,\quad x\ge 0,\quad y\ge 0
$$

左图里的灰色斜线就是：

$$
x+y=1
$$

图中编号 `1..8` 的点，是在这条约束线上取的 8 个 `p_current`。每个点都是一个合法概率分布：

$$
p_{\text{current}}^{(i)} = [x_i,\ 1-x_i]
$$

对每个 `p_current`，再计算 surprise 向量：

$$
s^{(i)} = -\log p_{\text{current}}^{(i)}
$$

也就是：

$$
s^{(i)} =
\left[
-\log x_i,\ -\log(1-x_i)
\right]
$$

这些 surprise 向量在图里用同色细虚线画出。

关键点：

```text
p_current 是概率向量，所以落在 x+y=1 上；
-log(p_current) 是 surprise 向量，不是概率向量，所以通常跑到约束线外。
```

红色星标是特殊点：

$$
p_{\text{current}} = p_{\text{target}}
$$

此时两个概率分布完全匹配，但：

$$
-\log p_{\text{current}}
$$

仍然不是概率向量，它不会和 $p_{\text{current}}$ 重合。

---

## 右图怎么看

右图横坐标是左图里的采样点编号：

$$
i = 1,2,\ldots,8
$$

纵坐标是：

$$
p_{\text{target}} \cdot [-\log p_{\text{current}}^{(i)}]
$$

也就是交叉熵：

$$
H\left(p_{\text{target}}, p_{\text{current}}^{(i)}\right)
= -\sum_k p_{\text{target}}(k)\log p_{\text{current}}^{(i)}(k)
$$

在二分类下展开：

$$
H(p_t,p_c)
= -p_t(1)\log p_c(1) - p_t(2)\log p_c(2)
$$

代入：

$$
p_t = [0.2,\ 0.8],\quad p_c=[x,\ 1-x]
$$

得到：

$$
H(x)
= -0.2\log x - 0.8\log(1-x)
$$

右图的曲线就是这件事：

$$
x \mapsto -0.2\log x - 0.8\log(1-x)
$$

红色虚线是最小值：

$$
H(p_t,p_t)
= -0.2\log 0.2 - 0.8\log 0.8
$$

$$
\approx 0.500
$$

注意这个最小值不是 0，而是目标分布自己的熵：

$$
H(p_t,p_t)=H(p_t)
$$

---

## 最凝练的公式理解

交叉熵：

$$
H(p_t,p_c)
= -\sum_k p_t(k)\log p_c(k)
$$

可以写成点积：

$$
\boxed{
H(p_t,p_c)
= p_t \cdot [-\log p_c]
}
$$

其中：

$$
-\log p_c(k)
$$

表示 current 对第 $k$ 个事件的 surprise。

所以：

$$
\boxed{
\text{交叉熵 = target 权重 对 current surprise 的加权平均}
}
$$

它不是：

$$
p_t \cdot p_c
$$

而是：

$$
p_t \cdot [-\log p_c]
$$

这就是为什么交叉熵会特别惩罚：

```text
target 很重要的位置，current 却给了很低概率。
```

因为当：

$$
p_c(k)\to 0
$$

时：

$$
-\log p_c(k)\to +\infty
$$

如果此时 $p_t(k)$ 又很大，那么这一项：

$$
-p_t(k)\log p_c(k)
$$

会变得很大。

---

## 和梯度的关系

如果：

$$
p_c = \text{softmax}(z)
$$

那么交叉熵对 logit 的梯度是：

$$
\boxed{
\frac{\partial H}{\partial z_k}
= p_c(k)-p_t(k)
}
$$

所以：

```text
loss 数值：target · surprise
梯度方向：current - target
```

这两个角度合起来非常漂亮：

1. **loss** 告诉你 current 对 target 平均有多惊讶。
2. **gradient** 告诉你每个 logit 应该往哪个方向改。

---

## 在 FastSAC C51 critic 中的位置

代码：

```python
critic_log_probs = F.log_softmax(q_outputs, dim=-1)
critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
```

对应数学：

$$
\text{critic\_log\_probs} = \log p_c
$$

$$
\text{target\_distributions} = p_t
$$

所以：

$$
\text{critic\_loss}
= -\sum_k p_t(k)\log p_c(k)
$$

也就是：

$$
\text{critic\_loss}
= p_t \cdot [-\log p_c]
$$

在 C51 里，每个 $k$ 对应一个固定 Q 原子：

$$
z_k \in [v_{\min}, v_{\max}]
$$

因此这行 loss 的含义是：

```text
Bellman projection 得到的 target Q 分布，
作为权重去检查 online critic 当前 Q 分布的 surprise。
```

如果 target 在某个 Q 原子上概率大，而 online critic 在这个原子上概率小，交叉熵会给出很大的 loss，并通过梯度把 online critic 的概率推回去。

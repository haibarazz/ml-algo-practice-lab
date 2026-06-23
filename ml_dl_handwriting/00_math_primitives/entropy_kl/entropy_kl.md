# Entropy And KL

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

离散熵：

$$H(p)=-\sum_i p_i\log p_i$$

KL 散度：

$$D_{KL}(p||q)=\sum_i p_i\log\frac{p_i}{q_i}$$

实现时要避免 `log(0)`，通常使用 `eps` 做裁剪。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    """TODO blank implementation."""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现无提示函数后直接运行测试区代码。

```python
def test_entropy_kl():
    assert np.allclose(entropy_kl([0.5, 0.5]), np.log(2))
    entropy, kl = entropy_kl([0.5, 0.5], [0.25, 0.75])
    assert np.allclose(entropy, np.log(2))
    assert kl > 0
    assert np.allclose(entropy_kl([1.0, 0.0]), 0.0, atol=1e-8)

test_entropy_kl()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    if q is None:
        return entropy
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sum(q)
    q_safe = np.clip(q, eps, 1.0)
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))
    return entropy, kl
```

### 解析

1. 输入先归一化，避免调用方给的是未归一化权重。
2. `eps` 裁剪用于避免 `log(0)`。
3. KL 不对称，`KL(p||q)` 和 `KL(q||p)` 通常不同。

## 工程要点 / 面试追问

### 核心公式

- $H(p)=-\sum_i p_i\log p_i$。
- $D_{KL}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}$，且 $H(p,q)=H(p)+D_{KL}(p\Vert q)$。

### 易错点

- 把 KL 当成距离使用，但 KL 不满足对称性，也不满足三角不等式。
- 忘记归一化概率，导致熵和 KL 没有概率意义。
- `0 log 0` 的极限是 0，但直接计算会得到 `nan`。
- KL 的方向很重要，`KL(p||q)` 和 `KL(q||p)` 对模式覆盖的偏好不同。

### 面试追问

::: details 参考回答：交叉熵、熵和 KL 散度之间是什么关系？

交叉熵可以分解为 `H(p, q) = H(p) + KL(p||q)`。训练时如果真实分布 p 固定，最小化交叉熵等价于最小化模型分布 q 到 p 的 KL。

:::

::: details 参考回答：为什么知识蒸馏和策略约束里经常出现 KL？

KL 用来度量两个分布的差异，因此适合表达“学生模型贴近教师模型”或“新策略不要偏离参考策略”。它不是普通距离，但在蒸馏、RLHF/DPO、策略约束中能自然刻画分布层面的偏移。

:::

::: details 参考回答：KL 为什么非负？什么时候等于 0？

KL 非负来自 Gibbs inequality，本质上是真实分布下的 log 概率比期望不会小于 0。只有当两个分布在支持集上几乎处处相等时，KL 才等于 0。

:::

::: details 参考回答：正向 KL 和反向 KL 在分布拟合上的直觉差异是什么？

正向 KL `KL(p||q)` 更怕 q 漏掉 p 有质量的区域，因此倾向覆盖多个模式。反向 KL `KL(q||p)` 更怕 q 放到 p 很小的区域，因此更容易 mode-seeking，只贴一个高密度模式。

:::

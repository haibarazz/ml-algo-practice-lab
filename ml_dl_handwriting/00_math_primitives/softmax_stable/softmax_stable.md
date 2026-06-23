# Softmax Stable

> Status: sample complete

## 手写实现约束

允许使用 `math` / NumPy；不允许调用任何框架内置 softmax。

## 原理最小说明

Softmax 把一组 logits 转成概率分布：

$$
softmax(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

直接对大 logits 求指数可能溢出。稳定写法是先减去最大值：

$$
softmax(x_i)=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m=\max_j x_j
$$

减去同一个常数不会改变 softmax 结果，但能显著降低溢出风险。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def softmax_stable_guided(x, axis=-1):
    """Compute numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)

    # TODO 1: subtract the max value along axis, keeping dims for broadcasting
    # shifted = ?

    # TODO 2: exponentiate shifted logits
    # exp_x = ?

    # TODO 3: normalize by the sum along axis
    # probs = ?

    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def softmax_stable(x, axis=-1):
    """Compute numerically stable softmax."""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现 `blank` 函数后直接运行：

```python
def test_softmax_stable():
    logits = np.array([1.0, 2.0, 3.0])
    probs = softmax_stable(logits)
    assert np.allclose(probs.sum(), 1.0)
    assert np.argmax(probs) == 2

    large_logits = np.array([1000.0, 1001.0, 1002.0])
    large_probs = softmax_stable(large_logits)
    expected = softmax_stable(np.array([0.0, 1.0, 2.0]))
    assert np.allclose(large_probs, expected)
    assert np.all(np.isfinite(large_probs))

    batch = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    batch_probs = softmax_stable(batch, axis=1)
    assert np.allclose(batch_probs.sum(axis=1), np.ones(2))
    assert np.allclose(batch_probs[1], np.array([1 / 3, 1 / 3, 1 / 3]))


test_softmax_stable()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np


def softmax_stable(x, axis=-1):
    """Compute numerically stable softmax."""
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
```

### 解析

1. `np.max(..., keepdims=True)` 保留维度，后续广播不会错。
2. `shifted = x - max(x)` 避免 `exp(1000)` 这类溢出。
3. softmax 的输出应该沿指定维度求和为 1。
4. 输入可以是一维 logits，也可以是二维 batch logits。

## 工程要点 / 面试追问

### 核心公式

- $\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$。
- 数值稳定写法：$\operatorname{softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}$。

### 易错点

- 忘记减最大值，遇到大 logits 会 overflow。
- `axis` 写死成最后一维，批量输入时不够通用。
- 没有 `keepdims=True`，二维或高维输入广播容易错。
- softmax 输出不是独立概率；所有类别概率相加必须为 1。

### 面试追问

::: details 参考回答：为什么减去同一个常数不会改变 softmax 结果？

softmax 的分子分母都会同时乘上 `exp(-c)`，这个公共因子会被约掉，所以减去同一个常数不改变结果。通常取最大值作为 `c`，可以让最大的指数项变成 1，避免 `exp` 上溢。

:::

::: details 参考回答：softmax 和 sigmoid 在二分类、多分类场景下有什么关系？

二分类时，两个 logits 的 softmax 可以化成对 logit 差值做 sigmoid，所以 sigmoid 可以看成二分类 softmax 的特例。多分类单标签场景用 softmax，因为类别之间互斥；多标签场景通常每个标签独立用 sigmoid。

:::

::: details 参考回答：为什么交叉熵里常把 log-softmax 和 NLL 合并计算？

分开算 softmax 再取 log 容易遇到上溢、下溢和 `log(0)`。把 log-softmax 和 NLL 合并后可以用 log-sum-exp 稳定计算，同时避免保存不必要的中间概率。

:::

::: details 参考回答：temperature 对 softmax 分布的尖锐程度有什么影响？

temperature 越小，logits 被放大，分布越尖锐，模型更偏向最大 logit。temperature 越大，分布越平滑，常用于蒸馏、采样多样性控制和校准分析。

:::

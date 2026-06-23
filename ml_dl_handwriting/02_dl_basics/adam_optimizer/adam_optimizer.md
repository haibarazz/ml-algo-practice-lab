# Adam Optimizer

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Adam 维护一阶矩和二阶矩：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

再做 bias correction 后更新参数。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def adam_optimizer(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
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

def adam_optimizer(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
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
def test_adam_optimizer():
    p = np.array([1.0, 2.0])
    g = np.array([0.1, -0.2])
    m = np.zeros(2); v = np.zeros(2)
    new_p, m, v = adam_optimizer(p, g, m, v, t=1, lr=0.001)
    assert new_p[0] < p[0]
    assert new_p[1] > p[1]
    assert np.all(m != 0)
    assert np.all(v > 0)

test_adam_optimizer()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def adam_optimizer(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param = np.asarray(param, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return new_param, m, v
```

### 解析

1. `m` 是梯度指数滑动平均。
2. `v` 是梯度平方指数滑动平均。
3. 第几步 `t` 用于 bias correction。
4. 更新方向由 `m_hat / sqrt(v_hat)` 决定。

## 工程要点 / 面试追问

### 核心公式

- $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$，$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$。
- $\hat m_t=m_t/(1-\beta_1^t)$，$\hat v_t=v_t/(1-\beta_2^t)$，$\theta_t=\theta_{t-1}-\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。

### 易错点

- 忘记 bias correction，训练早期步长会偏小。
- `t` 从 0 开始带入校正项会除零。
- `v` 应累计 `grad ** 2`，不是累计 `grad`。
- Adam 的 weight decay 和 L2 regularization 在自适应优化器中并不完全等价，AdamW 会解耦。

### 面试追问

- Adam 和 SGD momentum 的区别是什么？
- bias correction 为什么必要？
- AdamW 为什么把 weight decay 解耦？
- Adam 在稀疏梯度、噪声梯度场景下有什么优势和风险？

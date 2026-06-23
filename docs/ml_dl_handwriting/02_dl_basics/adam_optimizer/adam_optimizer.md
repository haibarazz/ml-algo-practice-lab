# Adam Optimizer

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/02_dl_basics/adam_optimizer/adam_optimizer.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: Adam 参数更新。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

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

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

见 `notes.md`。
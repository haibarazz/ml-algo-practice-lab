# Binary Logistic Regression

> Status: complete

## 题源线索

- Topic: 二分类逻辑回归。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

二分类逻辑回归：

$$p=\sigma(Xw+b)$$

使用 BCE 损失：

$$L=-\frac{1}{N}\sum y\log p+(1-y)\log(1-p)$$

梯度可写为 `X.T @ (p-y) / N`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def logistic_regression_binary(X, y, lr=0.1, steps=100):
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

def logistic_regression_binary(X, y, lr=0.1, steps=100):
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
def test_logistic_regression_binary():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    w, b, losses = logistic_regression_binary(X, y, lr=0.5, steps=200)
    assert losses[-1] < losses[0]
    probs = _sigmoid(X @ w + b)
    assert (probs[:2] < 0.5).all()
    assert (probs[2:] > 0.5).all()

test_logistic_regression_binary()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def logistic_regression_binary(X, y, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        eps = 1e-12
        losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        diff = p - y
        w -= lr * (X.T @ diff / X.shape[0])
        b -= lr * np.mean(diff)
    return w, b, losses
```

### 解析

1. sigmoid 输入做 clip，避免指数溢出。
2. BCE 里加 eps，避免 log(0)。
3. BCE + sigmoid 的梯度化简为 `p - y`。
4. 测试使用线性可分数据验证方向正确。

## 工程要点 / 面试追问

见 `notes.md`。

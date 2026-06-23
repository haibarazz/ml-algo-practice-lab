# Linear Regression GD

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/linear_regression_gd/linear_regression_gd.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: 线性回归 MSE 与梯度下降。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

线性回归预测：

$$\hat y = Xw + b$$

MSE 损失对参数的梯度：

$$
\nabla_w = \frac{2}{N}X^T(\hat y-y),\quad 
\nabla_b=\frac{2}{N}\sum_i(\hat y_i-y_i)$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def linear_regression_gd(X, y, lr=0.1, steps=100):
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

def linear_regression_gd(X, y, lr=0.1, steps=100):
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
def test_linear_regression_gd():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X[:, 0] + 1
    w, b, losses = linear_regression_gd(X, y, lr=0.05, steps=300)
    assert losses[-1] < losses[0]
    assert np.allclose(w[0], 2.0, atol=0.15)
    assert np.allclose(b, 1.0, atol=0.25)

test_linear_regression_gd()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def linear_regression_gd(X, y, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    losses = []
    for _ in range(steps):
        pred = X @ w + b
        diff = pred - y
        losses.append(np.mean(diff * diff))
        grad_w = 2.0 * X.T @ diff / X.shape[0]
        grad_b = 2.0 * np.mean(diff)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, losses
```

### 解析

1. 参数从零初始化即可。
2. 每步先 forward，再算梯度，再更新。
3. 学习率过大会震荡，过小会慢。
4. 测试不要求精确闭式解，只验证收敛到附近。


:::

## 工程要点 / 面试追问

见 `notes.md`。
# Binary Logistic Regression

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/logistic_regression_binary/logistic_regression_binary.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

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

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

### 核心公式

- $p(y=1|x)=\sigma(w^\top x+b)$，$\sigma(z)=\frac{1}{1+e^{-z}}$。
- $L=-[y\log p+(1-y)\log(1-p)]$，对 logits 的梯度为 $p-y$。

### 易错点

- 把 logistic regression 和线性回归混淆；LR 输出的是类别概率。
- sigmoid 直接计算可能溢出，BCE with logits 更稳定。
- BCE 梯度符号写反，参数更新方向会错。
- 默认阈值 0.5 不一定适合类别不均衡或不同业务成本。

### 面试追问

- 逻辑回归为什么是判别模型？
- 逻辑回归和线性回归的目标函数有什么区别？
- 为什么 BCE with logits 比 sigmoid 后再 BCE 更稳定？
- 二分类阈值如何根据 precision/recall 或业务成本调整？
# Multiclass Logistic Regression

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/logistic_regression_multiclass/logistic_regression_multiclass.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: 多分类 softmax regression。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

多分类逻辑回归也叫 softmax regression：

$$P(y=c|x)=softmax(xW+b)_c$$

交叉熵对 logits 的梯度是 `probs - one_hot(y)`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def logistic_regression_multiclass(X, y, num_classes, lr=0.1, steps=100):
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

def logistic_regression_multiclass(X, y, num_classes, lr=0.1, steps=100):
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
def test_logistic_regression_multiclass():
    X = np.array([[2, 0], [0, 2], [-2, -2]], dtype=float)
    y = np.array([0, 1, 2])
    W, b, losses = logistic_regression_multiclass(X, y, 3, lr=0.3, steps=200)
    probs = _softmax(X @ W + b)
    assert losses[-1] < losses[0]
    assert np.argmax(probs, axis=1).tolist() == [0, 1, 2]

test_logistic_regression_multiclass()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def logistic_regression_multiclass(X, y, num_classes, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    W = np.zeros((X.shape[1], num_classes))
    b = np.zeros(num_classes)
    losses = []
    for _ in range(steps):
        probs = _softmax(X @ W + b)
        losses.append(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))
        grad = probs
        grad[np.arange(len(y)), y] -= 1.0
        grad /= X.shape[0]
        W -= lr * (X.T @ grad)
        b -= lr * grad.sum(axis=0)
    return W, b, losses
```

### 解析

1. logits 形状是 `[N, C]`。
2. 先 softmax，再取目标类别概率计算 CE。
3. 梯度用 `probs - one_hot`。
4. 注意不要永久污染 `probs`，如果后面还要用原概率应 copy。


:::

## 工程要点 / 面试追问

见 `notes.md`。
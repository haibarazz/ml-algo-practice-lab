# Multiclass Logistic Regression

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/logistic_regression_multiclass/logistic_regression_multiclass.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

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

### 核心公式

- $p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}$，$z=XW+b$。
- $L=-\log p_{y}$，对 logits 的梯度为 $p-\operatorname{onehot}(y)$。

### 易错点

- 忘记稳定 softmax，logits 较大时容易 overflow。
- `grad = probs` 后原地修改，后续再用 `probs` 会被污染。
- W shape 写成 `[C,D]` 或 `[D,C]` 时要全程保持一致。
- one-vs-rest 和 multinomial softmax 的训练目标不同，不要混用结论。

### 面试追问

::: details 参考回答：softmax regression 和 one-vs-rest logistic regression 有什么区别？

softmax regression 一次性建模互斥多分类分布，所有类别概率共同归一化。one-vs-rest 会训练多个二分类器，类别之间不天然竞争，输出概率也不一定相互校准。

:::

::: details 参考回答：cross entropy 梯度为什么能化简为 `probs - onehot`？

softmax + cross entropy 对 logits 的梯度会化简为 `probs - onehot`。直觉上，模型给某类的概率高于目标就降低它，目标类概率低就提高它。

:::

::: details 参考回答：多分类类别不均衡时可以怎样加权？

可以在 cross entropy 中加入 class weight，让少数类或高成本类别的错误更重要。也可以重采样、使用 focal loss，或按业务指标调阈值，但要用验证集确认没有牺牲关键指标。

:::

::: details 参考回答：softmax 输出概率是否一定校准？如果不校准怎么办？

softmax 输出不一定校准，它只是相对分数归一化后的结果，深度模型常会过度自信。可以用 temperature scaling、Platt scaling、isotonic regression 或校准集后处理。

:::
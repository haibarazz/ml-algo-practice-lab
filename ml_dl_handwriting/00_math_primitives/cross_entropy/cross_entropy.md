# Cross Entropy

> Status: complete

## 题源线索

- Topic: 基于 logits 实现多分类交叉熵。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

多分类交叉熵常和 softmax 一起使用。对第 `i` 个样本：

$$L_i=-\log\frac{e^{z_{i,y_i}}}{\sum_c e^{z_{i,c}}}$$

为了数值稳定，先减去每行最大 logit，再计算 log-sum-exp。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def cross_entropy(logits, targets):
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

def cross_entropy(logits, targets):
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
def test_cross_entropy():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = np.array([0, 1])
    loss = cross_entropy(logits, targets)
    assert np.allclose(loss, 0.288725992, atol=1e-6)
    large = np.array([[1000.0, 1001.0]])
    assert np.isfinite(cross_entropy(large, np.array([1])))

test_cross_entropy()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(logits.shape[0]), targets]
    return np.mean(-correct + logsumexp)
```

### 解析

1. 减最大值避免 `exp(1000)` 溢出。
2. `correct` 取出目标类别对应的 shifted logit。
3. 每个样本损失是 `-correct + logsumexp`。
4. 最后对 batch 求平均。

## 工程要点 / 面试追问

见 `notes.md`。

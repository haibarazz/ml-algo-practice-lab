# LayerNorm Forward

> Status: complete

## 题源线索

- Topic: LayerNorm forward。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

LayerNorm 对每个样本的特征维做归一化：

$$\hat x=(x-\mu_{sample})/\sqrt{\sigma^2_{sample}+\epsilon}$$

和 BatchNorm 不同，它不依赖 batch 统计。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
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

def layer_norm_forward(X, gamma, beta, eps=1e-5):
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
def test_layer_norm_forward():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = layer_norm_forward(X, np.ones(3), np.zeros(3))
    assert np.allclose(out.mean(axis=1), np.zeros(2), atol=1e-6)
    assert np.allclose(out.var(axis=1), np.ones(2), atol=1e-5)

test_layer_norm_forward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def layer_norm_forward(X, gamma, beta, eps=1e-5):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    xhat = (X - mean) / np.sqrt(var + eps)
    return gamma * xhat + beta
```

### 解析

1. LayerNorm 沿最后一维统计。
2. gamma/beta 通常也是最后一维大小。
3. 它适合变长序列和小 batch 场景。

## 工程要点 / 面试追问

见 `notes.md`。

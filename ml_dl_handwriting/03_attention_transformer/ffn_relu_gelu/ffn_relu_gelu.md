# FFN Relu Gelu

> Status: complete

## 题源线索

- Topic: Transformer FFN with ReLU / GeLU。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Transformer FFN 是逐 token 的两层 MLP：

$$FFN(x)=W_2 activation(W_1x+b_1)+b_2$$

常见激活包括 ReLU 和 GeLU。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
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

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
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
def test_ffn_relu_gelu():
    X = np.array([[1.0, -1.0]])
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.ones((2, 1))
    b2 = np.array([0.0])
    assert np.allclose(ffn_relu_gelu(X, W1, b1, W2, b2, 'relu'), np.array([[1.0]]))
    out = ffn_relu_gelu(X, W1, b1, W2, b2, 'gelu')
    assert out.shape == (1, 1)

test_ffn_relu_gelu()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
    hidden = X @ W1 + b1
    if activation == 'relu':
        hidden = np.maximum(hidden, 0.0)
    elif activation == 'gelu':
        hidden = _gelu(hidden)
    else:
        raise ValueError("activation must be relu or gelu")
    return hidden @ W2 + b2
```

### 解析

1. FFN 对每个 token 独立作用。
2. ReLU 直接截断负值。
3. GeLU 常用 tanh 近似。
4. 最后一层投回模型维度。

## 工程要点 / 面试追问

见 `notes.md`。

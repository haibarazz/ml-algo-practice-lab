# Relu Sigmoid Tanh

> Status: complete

## 题源线索

- Topic: 激活函数及导数。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

常见激活函数：ReLU、sigmoid、tanh。面试常要求同时写 forward 和 derivative。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def relu_sigmoid_tanh(x):
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

def relu_sigmoid_tanh(x):
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
def test_relu_sigmoid_tanh():
    out = relu_sigmoid_tanh(np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(out["relu"], [0, 0, 1])
    assert np.allclose(out["relu_grad"], [0, 0, 1])
    assert np.allclose(out["sigmoid"][1], 0.5)
    assert np.allclose(out["tanh"][1], 0.0)

test_relu_sigmoid_tanh()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def relu_sigmoid_tanh(x):
    x = np.asarray(x, dtype=np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    tanh = np.tanh(x)
    relu = np.maximum(x, 0.0)
    return {
        "relu": relu,
        "relu_grad": (x > 0).astype(np.float64),
        "sigmoid": sigmoid,
        "sigmoid_grad": sigmoid * (1 - sigmoid),
        "tanh": tanh,
        "tanh_grad": 1 - tanh * tanh,
    }
```

### 解析

1. ReLU 梯度在 0 处不可导，本模块约定为 0。
2. sigmoid 梯度可写成 `s * (1-s)`。
3. tanh 梯度是 `1 - tanh^2`。

## 工程要点 / 面试追问

见 `notes.md`。

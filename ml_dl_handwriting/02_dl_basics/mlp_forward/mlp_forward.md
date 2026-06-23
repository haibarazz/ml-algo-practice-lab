# MLP Forward

> Status: complete

## 题源线索

- Topic: MLP 前向传播。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

两层 MLP 前向：

$$Z_1=XW_1+b_1$$
$$A_1=ReLU(Z_1)$$
$$Y=A_1W_2+b_2$$

本模块只做 forward，反向传播在 `mlp_backward`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
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

def mlp_forward(X, W1, b1, W2, b2):
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
def test_mlp_forward():
    X = np.array([[1.0, 2.0]])
    W1 = np.array([[1.0, -1.0], [0.5, 2.0]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0], [2.0]])
    b2 = np.array([0.1])
    out, cache = mlp_forward(X, W1, b1, W2, b2)
    assert np.allclose(cache["a1"], np.array([[2.0, 3.0]]))
    assert np.allclose(out, np.array([[8.1]]))

test_mlp_forward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    X = np.asarray(X, dtype=np.float64)
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    out = a1 @ W2 + b2
    return out, {"z1": z1, "a1": a1}
```

### 解析

1. 第一层线性变换后接 ReLU。
2. 第二层线性输出预测。
3. 返回 cache 是为了后续 backward 使用。

## 工程要点 / 面试追问

见 `notes.md`。

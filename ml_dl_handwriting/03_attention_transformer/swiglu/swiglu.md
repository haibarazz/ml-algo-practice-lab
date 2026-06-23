# SwiGLU

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

SwiGLU 使用门控前馈：

$$SwiGLU(x)= (SiLU(xW_g+b_g) \odot (xW_u+b_u))W_d+b_d$$

其中 `SiLU(z)=z*sigmoid(z)`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def swiglu(X, W_gate, b_gate, W_up, b_up, W_down, b_down):
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

def swiglu(X, W_gate, b_gate, W_up, b_up, W_down, b_down):
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
def test_swiglu():
    X = np.array([[1.0, 2.0]])
    Wg = np.ones((2, 3)); bg = np.zeros(3)
    Wu = np.ones((2, 3)); bu = np.zeros(3)
    Wd = np.ones((3, 1)); bd = np.zeros(1)
    out = swiglu(X, Wg, bg, Wu, bu, Wd, bd)
    assert out.shape == (1, 1)
    assert out[0, 0] > 0

test_swiglu()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def swiglu(X, W_gate, b_gate, W_up, b_up, W_down, b_down):
    gate = X @ W_gate + b_gate
    up = X @ W_up + b_up
    silu_gate = gate * _sigmoid(gate)
    return (silu_gate * up) @ W_down + b_down
```

### 解析

1. gate 分支经过 SiLU。
2. up 分支保持线性。
3. 两个分支逐元素相乘后 down projection。
4. 这是 LLaMA 系列常见 FFN 结构。

## 工程要点 / 面试追问

### 核心公式

- $SiLU(x)=x\sigma(x)$。
- $SwiGLU(x)=(SiLU(xW_g)\odot xW_u)W_d$。

### 易错点

- gate 和 up projection 的隐藏维度不一致。
- 用 sigmoid 直接代替 SiLU，变成普通 GLU 变体。
- 忘记 down projection，输出维度无法回到 `d_model`。
- SwiGLU 参数量通常比普通 FFN 大，hidden size 需要相应调整。

### 面试追问

- GLU、GeGLU、SwiGLU 的区别是什么？
- 为什么现代 LLM 常用 gated FFN？
- SwiGLU 相比 ReLU/GELU FFN 增加了哪些参数？
- SiLU gate 的输出为什么可以看作数据依赖的通道筛选？

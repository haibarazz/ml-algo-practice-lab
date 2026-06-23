# MLP Forward

> Status: complete

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

### 核心公式

- 两层 MLP：$h=\phi(XW_1+b_1)$，$\hat y=hW_2+b_2$。
- 如果 $\phi$ 是 ReLU，则 $\phi(z)=\max(0,z)$。

### 易错点

- bias 广播维度不清楚，导致单样本和 batch 输入表现不一致。
- 把 ReLU 放在第二层之后会改变题目定义。
- 不缓存中间值，后续 backward 难写。
- 多层线性层如果不加激活，本质仍等价于一个线性层。

### 面试追问

::: details 参考回答：为什么 MLP 需要非线性激活函数？

如果没有非线性，多层线性变换的复合仍然是一个线性变换，无法表达非线性决策边界。激活函数给网络引入分段线性或非线性结构，才让深层模型有更强表达能力。

:::

::: details 参考回答：多层线性层不加激活等价于什么？

多层线性层不加激活等价于一个单独的线性层，因为矩阵乘法可以合并。无论堆多少层，它的函数族仍是线性或仿射变换。

:::

::: details 参考回答：隐藏层宽度会怎样影响表达能力和过拟合风险？

隐藏层越宽，通常可表达的函数越复杂，欠拟合风险降低。宽度过大且数据不足时，模型可能记忆训练集，因此需要正则化、早停、dropout 或更多数据来控制泛化。

:::

::: details 参考回答：forward 中需要缓存哪些变量给 backward 使用？

至少要缓存输入 X、第一层线性输出 z1、激活 h，以及可能的参数引用。backward 需要这些中间值计算 ReLU mask、权重梯度和传回上一层的梯度。

:::

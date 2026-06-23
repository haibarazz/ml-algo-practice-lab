# Linear Regression GD

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

线性回归预测：

$$\hat y = Xw + b$$

MSE 损失对参数的梯度：

$$
\nabla_w = \frac{2}{N}X^T(\hat y-y),\quad 
\nabla_b=\frac{2}{N}\sum_i(\hat y_i-y_i)$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def linear_regression_gd(X, y, lr=0.1, steps=100):
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

def linear_regression_gd(X, y, lr=0.1, steps=100):
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
def test_linear_regression_gd():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2 * X[:, 0] + 1
    w, b, losses = linear_regression_gd(X, y, lr=0.05, steps=300)
    assert losses[-1] < losses[0]
    assert np.allclose(w[0], 2.0, atol=0.15)
    assert np.allclose(b, 1.0, atol=0.25)

test_linear_regression_gd()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def linear_regression_gd(X, y, lr=0.1, steps=100):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    losses = []
    for _ in range(steps):
        pred = X @ w + b
        diff = pred - y
        losses.append(np.mean(diff * diff))
        grad_w = 2.0 * X.T @ diff / X.shape[0]
        grad_b = 2.0 * np.mean(diff)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, losses
```

### 解析

1. 参数从零初始化即可。
2. 每步先 forward，再算梯度，再更新。
3. 学习率过大会震荡，过小会慢。
4. 测试不要求精确闭式解，只验证收敛到附近。

## 工程要点 / 面试追问

### 核心公式

- $\hat y=Xw+b$，$L=\frac{1}{n}\sum_i(\hat y_i-y_i)^2$。
- $\nabla_w L=\frac{2}{n}X^\top(Xw+b-y)$，$\nabla_b L=\frac{2}{n}\sum_i(\hat y_i-y_i)$。

### 易错点

- 梯度忘记除以 batch size，导致学习率含义变化。
- bias 梯度写成向量而不是标量或按输出维聚合。
- 学习率过大导致发散，过小导致收敛慢。
- 没有检查输入是否需要加截距项，闭式解和 GD 实现容易不一致。

### 面试追问

::: details 参考回答：线性回归的闭式解是什么？什么时候不适合直接用闭式解？

无正则且 `X^T X` 可逆时，闭式解是 `w = (X^T X)^(-1) X^T y`。特征维很高、样本很多、矩阵病态或需要在线训练时，直接求逆代价高且数值不稳，更适合 GD 或正则化解法。

:::

::: details 参考回答：GD、SGD、mini-batch SGD 的区别是什么？

GD 每次用全量数据算梯度，方向稳定但单步成本高。SGD 每次用单样本，噪声大但更新快；mini-batch SGD 在稳定性和计算效率之间折中，是深度学习里最常用的形式。

:::

::: details 参考回答：为什么特征标准化会影响梯度下降收敛速度？

特征尺度差异大时，损失曲面会很狭长，梯度下降在不同方向上的合适步长差别很大。标准化后曲面更接近各向同性，同一个学习率更容易稳定快速收敛。

:::

::: details 参考回答：L1/L2 正则会怎样改变目标函数和解的性质？

L2 正则会惩罚权重平方，使解更平滑、权重更小，通常不会产生严格稀疏。L1 正则惩罚绝对值，容易把部分权重压到 0，因此有特征选择效果。

:::

# SVM Hinge Loss

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

线性 SVM 的 hinge loss：

$$L=\frac{1}{2}\|w\|^2 + C\frac{1}{N}\sum_i \max(0, 1-y_i(w^Tx_i+b))$$

当 margin 小于 1 时，该样本贡献梯度。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def svm_hinge_loss(X, y, w, b=0.0, C=1.0):
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

def svm_hinge_loss(X, y, w, b=0.0, C=1.0):
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
def test_svm_hinge_loss():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([1.0, -1.0])
    loss, gw, gb = svm_hinge_loss(X, y, np.array([0.0, 0.0]), 0.0, C=1.0)
    assert np.allclose(loss, 1.0)
    assert np.allclose(gw, np.array([-0.5, 0.5]))
    assert np.allclose(gb, 0.0)

test_svm_hinge_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def svm_hinge_loss(X, y, w, b=0.0, C=1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    margins = 1 - y * (X @ w + b)
    active = margins > 0
    loss = 0.5 * np.dot(w, w) + C * np.mean(np.maximum(0, margins))
    grad_w = w - C * (X[active].T @ y[active]) / X.shape[0]
    grad_b = -C * np.sum(y[active]) / X.shape[0]
    return loss, grad_w, grad_b
```

### 解析

1. `margin = 1 - y * score`。
2. 只有 `margin > 0` 的样本进入 hinge 梯度。
3. 正则项梯度是 `w`。
4. 标签约定必须是 `{-1, +1}`。

## 工程要点 / 面试追问

### 核心公式

- 二分类 hinge loss：$L=\max(0,1-y(w^\top x+b))$，其中 $y\in\{-1,+1\}$。
- 带 L2 正则的软间隔目标常写为 $\frac{1}{2}\lVert w\rVert^2+C\sum_i\xi_i$。

### 易错点

- 标签用 0/1 而不是 -1/+1，会让 margin 公式失效。
- 忘记正则项，无法体现最大间隔。
- active mask 条件写反；只有 margin 小于 1 的样本贡献 hinge 梯度。
- hinge loss 在折点不可导，通常使用次梯度。

### 面试追问

::: details 参考回答：SVM 的 margin 几何意义是什么？

margin 是样本到决策超平面的带符号距离乘以标签，表示分类不仅要对，还要离边界足够远。最大化 margin 可以得到更稳健的决策边界，这是 SVM 泛化直觉的核心。

:::

::: details 参考回答：hinge loss 和 logistic loss 的差别是什么？

hinge loss 在 margin 大于 1 后损失为 0，只关注分类错误或离边界不够远的样本。logistic loss 对所有样本都有非零惩罚，并输出概率意义更强，但没有硬间隔的几何解释那么直接。

:::

::: details 参考回答：C 参数变大或变小会怎样影响间隔和误分类？

C 越大，模型越重视训练误分类，间隔可能变窄，过拟合风险更高。C 越小，模型更强调大间隔和正则化，允许更多训练错误，偏差可能更大。

:::

::: details 参考回答：核 SVM 为什么在大样本上训练代价高？

核 SVM 需要维护样本之间的核矩阵或大量支持向量，训练复杂度和存储都会随样本数快速增长。大样本下预测也可能慢，因为每次预测要和许多支持向量计算核函数。

:::

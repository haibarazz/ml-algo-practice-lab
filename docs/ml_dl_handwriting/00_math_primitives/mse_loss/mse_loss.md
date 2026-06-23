# MSE Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/00_math_primitives/mse_loss/mse_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

MSE 衡量预测值和真实值的平方差平均：

$$MSE=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$$

它常用于回归，也常作为手写反向传播的入门损失。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def mse_loss(y_true, y_pred):
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

def mse_loss(y_true, y_pred):
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
def test_mse_loss():
    assert np.allclose(mse_loss([1, 2, 3], [1, 2, 5]), 4 / 3)
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = y + 1.0
    assert np.allclose(mse_loss(y, pred), 1.0)

test_mse_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    return np.mean(diff * diff)
```

### 解析

1. 先统一转换为浮点数组。
2. 按所有元素求平均，而不是只按 batch 维。
3. 如果后续写 backward，梯度分母要和这里的 mean 定义一致。


:::

## 工程要点 / 面试追问

### 核心公式

- $MSE=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$。
- 若对预测值求梯度：$\frac{\partial MSE}{\partial \hat y_i}=\frac{2}{N}(\hat y_i-y_i)$。

### 易错点

- sum 和 mean 混用会导致 loss 与梯度尺度不一致。
- `y_true - y_pred` 和 `y_pred - y_true` 对 loss 无影响，但对梯度方向有影响。
- 多维输出时要说清楚是按所有元素平均，还是先按样本再按输出维聚合。
- MSE 对异常值敏感，数据噪声重尾时不一定是好选择。

### 面试追问

::: details 参考回答：MSE 为什么比 MAE 更容易受异常值影响？

MSE 对误差做平方，误差变大时惩罚是二次增长；MAE 是线性增长。异常值往往误差很大，所以在 MSE 下会主导 loss 和梯度。

:::

::: details 参考回答：MSE、RMSE、MAE 分别适合什么回归评价场景？

MSE 适合噪声接近高斯、希望大误差被更强惩罚的回归训练。RMSE 和原标签同量纲，便于解释；MAE 对异常值更稳健，适合重尾噪声或更关心绝对偏差的场景。

:::

::: details 参考回答：如果训练时用 MSE，梯度分母应该除以 batch size 还是元素总数？

分母要和 loss 的定义一致：如果 loss 是所有元素的 mean，梯度就除以元素总数；如果先按样本聚合再平均，就除以 batch size。面试里关键不是固定答案，而是 loss 数值和梯度尺度必须一致。

:::

::: details 参考回答：Huber loss 如何在 MSE 和 MAE 之间折中？

Huber loss 在小误差区间使用平方惩罚，保持 MSE 的平滑梯度；在大误差区间切换成近似线性惩罚，降低异常值影响。阈值 delta 控制从 MSE 到 MAE 的切换点。

:::
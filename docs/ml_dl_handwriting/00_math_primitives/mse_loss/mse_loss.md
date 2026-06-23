# MSE Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/00_math_primitives/mse_loss/mse_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: 实现均方误差。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

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

见 `notes.md`。
# RMSNorm Forward

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/02_dl_basics/rms_norm_forward/rms_norm_forward.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

RMSNorm 不减均值，只按均方根归一化：

$$y=x/\sqrt{mean(x^2)+\epsilon}\cdot \gamma$$

它比 LayerNorm 少了中心化步骤。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def rms_norm_forward(X, gamma, eps=1e-6):
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

def rms_norm_forward(X, gamma, eps=1e-6):
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
def test_rms_norm_forward():
    X = np.array([[3.0, 4.0], [1.0, 1.0]])
    out = rms_norm_forward(X, np.ones(2), eps=0.0)
    assert np.allclose(np.sqrt(np.mean(out * out, axis=1)), np.ones(2))

test_rms_norm_forward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def rms_norm_forward(X, gamma, eps=1e-6):
    X = np.asarray(X, dtype=np.float64)
    rms = np.sqrt(np.mean(X * X, axis=-1, keepdims=True) + eps)
    return X / rms * gamma
```

### 解析

1. RMS 是平方均值再开方。
2. 不减均值，因此输出均值不一定为 0。
3. LLaMA 等模型常用 RMSNorm 变体。


:::

## 工程要点 / 面试追问

### 核心公式

- $RMS(x)=\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}$。
- $y=\gamma\frac{x}{RMS(x)}$；RMSNorm 去掉了 LayerNorm 的减均值步骤。

### 易错点

- 写成 LayerNorm，额外减了均值。
- 沿 batch 维求 RMS，而不是沿最后的 feature 维。
- 忘记 gamma 或 gamma shape 不能广播。
- RMSNorm 只保留 re-scaling，不提供 re-centering invariance。

### 面试追问

- RMSNorm 相比 LayerNorm 省掉了什么计算？
- 为什么很多 LLM 使用 RMSNorm？
- RMSNorm 保留了什么不变性，又失去了什么不变性？
- RMSNorm 的 eps 应该放在哪里？
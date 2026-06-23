# RoPE Rotate Half

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/02_model_architecture/rope_rotate_half/rope_rotate_half.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `model/model_minimind.py:62-84`

## 手写实现约束

允许 NumPy；只实现单个向量或二维数组最后一维旋转。

## 原理最小说明

MiniMind 的 RoPE 先把最后一维拆成两半，`rotate_half([a,b,c,d])=[-c,-d,a,b]`，再计算 `x*cos + rotate_half(x)*sin`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def apply_rope_rotate_half(x, cos, sin):
    """TODO guided implementation."""
    # TODO 1: 实现 rotate_half
    # TODO 2: 确保 cos/sin 可广播
    # TODO 3: 返回 x*cos + rotate_half(x)*sin
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def apply_rope_rotate_half(x, cos, sin):
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
import numpy as np


def test_apply_rope_rotate_half():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cos = np.ones(4)
    sin = np.zeros(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), x)
    cos = np.zeros(4)
    sin = np.ones(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), np.array([-3.0, -4.0, 1.0, 2.0]))


test_apply_rope_rotate_half()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def apply_rope_rotate_half(x, cos, sin):
    x = np.asarray(x, dtype=np.float64)
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.asarray(cos) + rotated * np.asarray(sin)
```

### 解析

1. 有些实现按偶奇维成对旋转，MiniMind 这里是 half split 形式。
2. cos/sin 来自预计算 position embedding。


:::

## 工程要点 / 面试追问

### Source Mapping

- `model/model_minimind.py:62-84`

### 常见坑

- 有些实现按偶奇维成对旋转，MiniMind 这里是 half split 形式。
- cos/sin 来自预计算 position embedding。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？
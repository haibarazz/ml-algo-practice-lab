# SwiGLU Feed Forward

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/02_model_architecture/swiglu_feed_forward/swiglu_feed_forward.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `model/model_minimind.py:136-146`

## 手写实现约束

允许 NumPy；矩阵乘用 `@`。

## 原理最小说明

MiniMind 的 FFN 是 SwiGLU 风格：`down(silu(gate(x)) * up(x))`。gate 分支决定保留哪些通道，up 分支提供内容。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def swiglu_ffn(x, w_gate, w_up, w_down):
    """TODO guided implementation."""
    # TODO 1: 实现 silu
    # TODO 2: 计算 gate/up 两条分支
    # TODO 3: 逐元素相乘
    # TODO 4: 再乘 down 投影
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def swiglu_ffn(x, w_gate, w_up, w_down):
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


def test_swiglu_ffn():
    x = np.array([[1.0, 2.0]])
    w_gate = np.eye(2)
    w_up = np.eye(2)
    w_down = np.eye(2)
    out = swiglu_ffn(x, w_gate, w_up, w_down)
    expected = x * (x / (1 + np.exp(-x)))
    assert np.allclose(out, expected)


test_swiglu_ffn()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def swiglu_ffn(x, w_gate, w_up, w_down):
    x = np.asarray(x, dtype=np.float64)
    gate = x @ np.asarray(w_gate, dtype=np.float64)
    up = x @ np.asarray(w_up, dtype=np.float64)
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ np.asarray(w_down, dtype=np.float64)
```

### 解析

1. 如果 hidden_act 换掉，gate 分支的激活也会变。
2. 三个矩阵 shape 要能完成 hidden -> intermediate -> hidden。


:::

## 工程要点 / 面试追问

### Source Mapping

- `model/model_minimind.py:136-146`

### 常见坑

- 如果 hidden_act 换掉，gate 分支的激活也会变。
- 三个矩阵 shape 要能完成 hidden -> intermediate -> hidden。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？
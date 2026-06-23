# RMSNorm in MiniMind

> Status: complete

## Source Mapping

- `model/model_minimind.py:50-60`

## 手写实现约束

允许 NumPy；不调用框架归一化层。

## 原理最小说明

RMSNorm 不减均值，只按最后一维的均方根缩放，再乘可学习 weight。它比 LayerNorm 少一个均值中心化步骤。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def rms_norm(x, weight, eps=1e-6):
    """TODO guided implementation."""
    # TODO 1: 沿最后一维求 mean(x^2)
    # TODO 2: 加 eps 后 rsqrt
    # TODO 3: 乘回 x
    # TODO 4: 乘 weight 广播
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def rms_norm(x, weight, eps=1e-6):
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


def test_rms_norm():
    x = np.array([[3.0, 4.0], [0.0, 2.0]])
    weight = np.array([1.0, 2.0])
    out = rms_norm(x, weight, eps=0.0)
    assert np.allclose(out[0], np.array([0.84852814, 2.2627417]), atol=1e-6)
    assert np.allclose(out[1], np.array([0.0, 2.82842712]), atol=1e-6)


test_rms_norm()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def rms_norm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight
```

### 解析

1. RMSNorm 的 eps 是数值稳定项。
2. weight shape 通常等于 hidden_size。

## 工程要点 / 面试追问

见 `notes.md`。

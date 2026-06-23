# Repeat KV for GQA

> Status: complete

## Source Mapping

- `model/model_minimind.py:86-89`
- `model/model_minimind.py:91-134`

## 手写实现约束

允许 NumPy；输入 shape 为 `[B, S, H_kv, D]`。

## 原理最小说明

GQA 中 Q head 数量大于 KV head。计算 attention 前，需要把每个 KV head 复制 `n_rep=num_q_heads/num_kv_heads` 次。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def repeat_kv(x, n_rep):
    """TODO guided implementation."""
    # TODO 1: n_rep 为 1 时直接返回
    # TODO 2: 在 head 维后插入 repeat 维
    # TODO 3: reshape 成重复后的 head 数量
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def repeat_kv(x, n_rep):
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


def test_repeat_kv():
    x = np.array([[[[1], [2]], [[3], [4]]]])
    out = repeat_kv(x, 2)
    assert out.shape == (1, 2, 4, 1)
    assert out[0, 0, :, 0].tolist() == [1, 1, 2, 2]
    assert repeat_kv(x, 1).shape == x.shape


test_repeat_kv()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return np.repeat(x[:, :, :, None, :], n_rep, axis=3).reshape(b, s, h * n_rep, d)
```

### 解析

1. 这里重复的是 KV，不是 Q。
2. 真实 attention 前还会 transpose 到 `[B, H, S, D]`。

## 工程要点 / 面试追问

见 `notes.md`。

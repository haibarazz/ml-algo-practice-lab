# Cosine LR Schedule

> Status: complete

## Source Mapping

- `trainer/trainer_utils.py:40-41`

## 手写实现约束

只用 math。

## 原理最小说明

MiniMind 的学习率不是从 lr 衰减到 0，而是按 `lr*(0.1 + 0.45*(1+cos(pi*step/total)))` 从 1.0lr 平滑到 0.1lr。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math


def get_lr(current_step, total_steps, lr):
    """TODO guided implementation."""
    # TODO 1: 计算 step/total 的 cosine phase
    # TODO 2: 代入 0.1+0.45*(1+cos)
    # TODO 3: 返回缩放后的 lr
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import math


def get_lr(current_step, total_steps, lr):
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
import math


def test_get_lr():
    assert abs(get_lr(0, 100, 1.0) - 1.0) < 1e-12
    assert abs(get_lr(100, 100, 1.0) - 0.1) < 1e-12
    assert abs(get_lr(50, 100, 2.0) - 1.1) < 1e-12


test_get_lr()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import math

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### 解析

1. 没有 warmup；这是项目当前实现，不是通用最佳实践。
2. total_steps 不能为 0。

## 工程要点 / 面试追问

见 `notes.md`。

# 余弦学习率：从初始学习率平滑衰减

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/03_pretraining/cosine_lr_schedule/cosine_lr_schedule.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


复刻 `trainer_utils.get_lr` 中 MiniMind 使用的 cosine schedule。

## 学习目标

- 理解每个 step 动态更新 optimizer learning rate。
- 掌握 MiniMind 的余弦公式和衰减范围。
- 理解学习率调度和训练稳定性的关系。

## MiniMind 源码定位

- `trainer/trainer_utils.py:40-41`
- `trainer/train_pretrain.py:24-80`

## 源码机制详解

`train_epoch` 每个 step 都调用 `get_lr(epoch * iters + step, total_steps, learning_rate)`，然后把结果写回 optimizer 的每个 param group。也就是说学习率不是每个 epoch 变一次，而是按全局 step 连续变化。
MiniMind 的公式是 `lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))`。当 step 为 0 时系数为 1.0；当 step 到 total_steps 时系数为 0.1，因此它从初始学习率平滑降到 10%。
这个实现没有 warmup，适合小模型和简化训练脚本。读大模型训练代码时，常见变体是 warmup + cosine decay，本模块先练最核心的 cosine 部分。

## 关键公式与数据流

- $lr_t=lr_0\left(0.1+0.45(1+\cos(\pi t/T))\right)$。
- $t=0$ 时 $lr_t=lr_0$；$t=T$ 时 $lr_t=0.1lr_0$。

## 为什么练这个

- 手写学习率函数能让你读懂训练日志里的 lr 变化。
- 这个模块也解释为什么 train_epoch 需要知道 epoch、iters 和 step。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math


def get_lr(current_step, total_steps, lr):
    """带提示实现。"""
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
    """无提示实现。"""
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

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import math

def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### 解析

1. 没有 warmup；这是项目当前实现，不是通用最佳实践。
2. total_steps 不能为 0。


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- $lr_t=lr_0\left(0.1+0.45(1+\cos(\pi t/T))\right)$。
- $t=0$ 时 $lr_t=lr_0$；$t=T$ 时 $lr_t=0.1lr_0$。

### 易错点

- current_step 和 total_steps 如果从不同基准计数，会导致衰减过快或过慢。
- total_steps 为 0 时需要避免除零。
- 只在 epoch 开头更新 lr，会和源码行为不同。

### 面试追问

::: details 参考回答：为什么训练中常用余弦学习率衰减？

它前期保持较大学习率以快速下降，后期平滑降低步长以减少震荡。相比阶梯衰减，余弦曲线没有突变，更适合长训练过程。

:::

::: details 参考回答：MiniMind 这个公式最低为什么不是 0？

最低保留 10% 初始学习率可以避免训练后期完全停滞。对小规模训练来说，这是一种简单保守的衰减策略。

:::
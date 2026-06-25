# 梯度累积：用小 batch 模拟大 batch

拆解 MiniMind 训练循环里 loss 缩放、反向传播和 optimizer step 的计数逻辑。

## 学习目标

- 理解为什么 loss 要除以 accumulation_steps。
- 掌握每隔 N 个 mini-batch 才执行 optimizer.step 的逻辑。
- 理解 epoch 末尾剩余 batch 也需要补一次 step。

## MiniMind 源码定位

- `trainer/train_pretrain.py:24-80`
- `trainer/train_full_sft.py:24-80`

## 源码机制详解

MiniMind 每个 mini-batch 都 forward/backward，但不是每次都更新参数。它先把 `loss = loss / accumulation_steps`，然后 `backward()` 累积梯度；只有当 `step % accumulation_steps == 0` 时才 unscale、clip、optimizer.step、zero_grad。
loss 除以 accumulation_steps 是为了让累积 N 次后的梯度均值接近大 batch 一次 backward 的梯度。如果不除，等效学习率会放大 N 倍。
函数末尾还有一个补偿逻辑：如果最后一个 step 不是 accumulation_steps 的整数倍，说明梯度已经累积但还没更新，需要执行一次 optimizer step。这个细节对小数据集或恢复训练很重要。

## 关键公式与数据流

- 等效 batch size：$B_{eff}=B_{micro}\times accumulation\_steps\times world\_size$。
- 每个 micro-batch 使用 $L/N$ 反传，累积 N 次后得到平均梯度。

## 为什么练这个

- 这个练习把训练循环从“调库”拆成可数的状态机。
- 它对应真实训练中显存不够但想增大有效 batch 的常见方案。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def accumulation_plan(num_batches, accumulation_steps):
    """带提示实现。"""
    # TODO 1: 遍历 batch step 从 1 开始
    # TODO 2: step 可整除时记录 optimizer step
    # TODO 3: 末尾剩余 batch 时补一次
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def accumulation_plan(num_batches, accumulation_steps):
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
def test_accumulation_plan():
    assert accumulation_plan(8, 4) == [4, 8]
    assert accumulation_plan(10, 4) == [4, 8, 10]
    assert accumulation_plan(3, 8) == [3]


test_accumulation_plan()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def accumulation_plan(num_batches, accumulation_steps):
    updates = []
    for step in range(1, num_batches + 1):
        if step % accumulation_steps == 0:
            updates.append(step)
    if num_batches > 0 and num_batches % accumulation_steps != 0:
        updates.append(num_batches)
    return updates
```

### 解析

1. 真实训练中 backward 每个 batch 都做，optimizer step 不是每个 batch 都做。
2. loss 除以 accumulation_steps 是为了保持梯度尺度。

## 工程要点 / 面试追问

### 关键公式与数据流

- 等效 batch size：$B_{eff}=B_{micro}\times accumulation\_steps\times world\_size$。
- 每个 micro-batch 使用 $L/N$ 反传，累积 N 次后得到平均梯度。

### 易错点

- 忘记除以 accumulation_steps，会让梯度尺度变大。
- 忘记处理尾部剩余 batch，会丢掉最后几步梯度。
- zero_grad 放错位置会清空尚未累积完的梯度。

### 面试追问

::: details 参考回答：梯度累积和直接增大 batch size 完全等价吗？

在没有 BatchNorm、dropout 随机性和分布式同步差异时，梯度均值接近等价。但优化器状态更新频率、随机性和日志 step 语义仍可能不同。

:::

::: details 参考回答：为什么梯度裁剪要放在 unscale 之后？

混合精度下梯度可能被 scaler 放大，直接裁剪会裁到错误尺度。先 unscale 再 clip，裁剪阈值才对应真实梯度范数。

:::

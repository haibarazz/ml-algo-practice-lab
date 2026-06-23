# Gradient Accumulation Counter

> Status: complete

## Source Mapping

- `trainer/train_pretrain.py:24-80`
- `trainer/train_full_sft.py:24-80`

## 手写实现约束

只用 Python list；不做真实反向传播。

## 原理最小说明

梯度累积把 loss 除以 accumulation_steps，每隔若干 step 执行一次 optimizer step。最后不足一个完整累积窗口，也要补一次 step。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def accumulation_plan(num_batches, accumulation_steps):
    """TODO guided implementation."""
    # TODO 1: 遍历 batch step 从 1 开始
    # TODO 2: step 可整除时记录 optimizer step
    # TODO 3: 末尾剩余 batch 时补一次
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def accumulation_plan(num_batches, accumulation_steps):
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
def test_accumulation_plan():
    assert accumulation_plan(8, 4) == [4, 8]
    assert accumulation_plan(10, 4) == [4, 8, 10]
    assert accumulation_plan(3, 8) == [3]


test_accumulation_plan()
print("All tests passed.")
```

## STOP HERE

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

见 `notes.md`。

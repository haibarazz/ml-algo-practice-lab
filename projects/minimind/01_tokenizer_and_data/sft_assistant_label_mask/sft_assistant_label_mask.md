# SFT 标签掩码：只训练 assistant 回复

拆解 `SFTDataset.generate_labels` 如何在多轮对话中只保留 assistant span 的监督信号。

## 学习目标

- 理解 chat template 把 role/content/tools 转成单条训练字符串的过程。
- 掌握 assistant 起止 token 如何决定 label mask 边界。
- 理解为什么 SFT 不应该训练 user/system token。

## MiniMind 源码定位

- `dataset/lm_dataset.py:58-119`
- `trainer/train_full_sft.py:24-80`

## 源码机制详解

`SFTDataset` 先通过 tokenizer 的 `apply_chat_template` 把 conversations 渲染成模型真正看到的 prompt。这个过程会插入 `<|im_start|>role`、`<|im_end|>`、tool call、think 标签等特殊结构。
`generate_labels` 初始化全 `-100`，然后扫描 `bos_id = tokenizer('<|im_start|>assistant\n')`。一旦找到 assistant 开始位置，就一直走到 `eos_id = tokenizer('<|im_end|>\n')`，只把这段 token 复制到 labels 中。
训练脚本仍然调用同一个 `model(input_ids, labels=labels)`。区别在于 labels 大部分位置都是 `-100`，因此 loss 只来自 assistant 回复。这样模型学习“在给定 system/user/tool 上下文时生成 assistant”，而不是学习复述用户输入。

## 关键公式与数据流

- SFT 目标：$L=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log p_\theta(x_t|x_{<t})$。
- $\mathcal A$ 是 assistant token 位置集合；非 assistant、pad、system、user 位置 label 设为 $-100$。

## 为什么练这个

- 手写 mask 能训练你从模板文本中恢复监督边界。
- 这个能力直接对应真实 SFT 数据清洗和 chat template 对齐问题。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
    """带提示实现。"""
    # TODO 1: 初始化全 -100 labels
    # TODO 2: 扫描 assistant_bos 子序列
    # TODO 3: 找到后续 assistant_eos 子序列
    # TODO 4: 只复制回复 span 到 labels
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
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
def test_generate_assistant_labels():
    ids = [9, 101, 102, 5, 6, 201, 7, 101, 102, 8, 201, 0]
    labels = generate_assistant_labels(ids, assistant_bos=[101, 102], assistant_eos=[201], max_length=len(ids))
    assert labels == [-100, -100, -100, 5, 6, 201, -100, -100, -100, 8, 201, -100]


test_generate_assistant_labels()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def generate_assistant_labels(input_ids, assistant_bos, assistant_eos, max_length):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(assistant_bos)] == assistant_bos:
            start = i + len(assistant_bos)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(assistant_eos)] == assistant_eos:
                    break
                end += 1
            stop = min(end + len(assistant_eos), max_length)
            for j in range(start, stop):
                labels[j] = input_ids[j]
            i = stop
        else:
            i += 1
    return labels
```

### 解析

1. 真实项目中的 assistant_bos/eos 来自 tokenizer 编码。
2. 如果 eos 被截断，labels 会延伸到 max_length。

## 工程要点 / 面试追问

### 关键公式与数据流

- SFT 目标：$L=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log p_\theta(x_t|x_{<t})$。
- $\mathcal A$ 是 assistant token 位置集合；非 assistant、pad、system、user 位置 label 设为 $-100$。

### 易错点

- assistant 起始 token 少了换行，mask 会完全扫不到。
- 把 user/system 也计入 loss，会稀释回答部分监督，还会鼓励模型复述输入。
- 不同 chat template 的角色边界不同，不能复用硬编码 token 序列。

### 面试追问

::: details 参考回答：SFT 为什么通常只对 assistant token 计 loss？

部署时模型的任务是在 prompt 条件下生成 assistant 回复，prompt 是条件而不是目标。只训练 assistant token 能把梯度集中到回答行为上，避免模型学习复述用户输入。

:::

::: details 参考回答：多轮对话里 label mask 的边界最容易错在哪里？

最容易错在 role special token 和换行。真实 tokenizer 看到的是模板渲染后的 token 序列，必须用 tokenizer 编码出来的 assistant 起止片段匹配，而不是靠字符串长度估算。

:::

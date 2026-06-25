# 预训练样本打包：BOS/EOS、pad 与 label mask

复刻 `PretrainDataset` 怎样把一条普通文本变成 causal LM 训练样本。

## 学习目标

- 理解预训练样本中 `input_ids` 和 `labels` 为什么几乎相同。
- 理解 BOS/EOS/pad token 在固定长度 batch 中的作用。
- 掌握 `-100` 如何让 PyTorch cross entropy 忽略 padding。

## MiniMind 源码定位

- `dataset/lm_dataset.py:37-55`
- `model/model_minimind.py:245-253`

## 源码机制详解

`PretrainDataset.__getitem__` 先取出 `sample['text']`，调用 tokenizer 得到内容 token，并为 BOS 和 EOS 预留两个位置。随后它拼接 `[BOS] + tokens + [EOS]`，再用 pad token 补齐到 `max_length`。
返回的 `labels` 是 `input_ids` 的拷贝，但 pad 位置被改成 `-100`。这是因为 MiniMind 的 loss 在模型内部做 shift：logits 去掉最后一位，labels 去掉第一位，所以数据集不需要自己构造 `x/y` 两份序列。
这个设计把“序列对齐”集中放在模型 loss 中，把“样本定长化”和“无效位置屏蔽”放在 dataset 中。读训练脚本时看到 `model(input_ids, labels=labels)`，就知道 shift 和 ignore 都在后续完成。

## 关键公式与数据流

- 打包：$ids=[BOS] + tokenizer(text)[:L-2] + [EOS] + [PAD]\times(L-|ids|)$。
- 标签：$labels_i=ids_i$，若 $ids_i=PAD$，则 $labels_i=-100$。
- 模型内 shift：用 $logits_{t}$ 预测 $labels_{t+1}$。

## 为什么练这个

- 手写这个函数能把“文本样本”和“语言模型训练样本”的差异讲清楚。
- 测试关注截断、补齐、BOS/EOS 和 pad label，这是实际训练里最常见的数据 bug。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
    """带提示实现。"""
    # TODO 1: 截断内容 token，给 BOS/EOS 留位置
    # TODO 2: 拼接 BOS 和 EOS
    # TODO 3: pad 到 max_length
    # TODO 4: labels 中 pad 位置改成 -100
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
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
def test_pack_pretrain_tokens():
    input_ids, labels = pack_pretrain_tokens([7, 8, 9, 10], bos_id=1, eos_id=2, pad_id=0, max_length=5)
    assert input_ids == [1, 7, 8, 9, 2]
    assert labels == [1, 7, 8, 9, 2]
    input_ids, labels = pack_pretrain_tokens([7], bos_id=1, eos_id=2, pad_id=0, max_length=5)
    assert input_ids == [1, 7, 2, 0, 0]
    assert labels == [1, 7, 2, -100, -100]


test_pack_pretrain_tokens()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
    content_len = max_length - 2
    content = list(tokens)[:content_len]
    input_ids = [bos_id] + content + [eos_id]
    input_ids = input_ids + [pad_id] * (max_length - len(input_ids))
    labels = [tok if tok != pad_id else -100 for tok in input_ids]
    return input_ids, labels
```

### 解析

1. 注意 max_length 必须至少容纳 BOS/EOS。
2. labels 忽略 pad，input_ids 仍保留 pad。

## 工程要点 / 面试追问

### 关键公式与数据流

- 打包：$ids=[BOS] + tokenizer(text)[:L-2] + [EOS] + [PAD]\times(L-|ids|)$。
- 标签：$labels_i=ids_i$，若 $ids_i=PAD$，则 $labels_i=-100$。
- 模型内 shift：用 $logits_{t}$ 预测 $labels_{t+1}$。

### 易错点

- 忘记给 BOS/EOS 留位置，会导致末尾 EOS 被截断。
- labels 不屏蔽 pad，会让模型学习预测 padding。
- 在 dataset 和 model 两边重复 shift，会产生错位两格的隐蔽错误。

### 面试追问

::: details 参考回答：为什么预训练时 `input_ids` 和 `labels` 可以先设成一样？

因为 causal LM 的错位预测在模型 loss 中完成：`logits[..., :-1, :]` 对齐 `labels[..., 1:]`。数据集只需要提供完整 token 序列和无效位置 mask。

:::

::: details 参考回答：为什么 pad label 用 `-100` 而不是 pad token id？

`torch.nn.functional.cross_entropy` 默认用 `ignore_index=-100` 跳过这些位置。若使用 pad token id，pad 会成为一个真实监督目标，长短样本的 padding 数量还会改变 loss 权重。

:::

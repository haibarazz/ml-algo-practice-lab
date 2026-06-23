# Pretrain Token Pack

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/01_tokenizer_and_data/pretrain_token_pack/pretrain_token_pack.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `dataset/lm_dataset.py:37-55`

## 手写实现约束

只用 Python list；不调用 tokenizer。

## 原理最小说明

预训练样本把文本 token 包成 `[BOS] + tokens + [EOS] + padding`。labels 复制 input_ids，但 padding 位置置为 `-100`，这样 loss 会忽略 padding。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def pack_pretrain_tokens(tokens, bos_id, eos_id, pad_id, max_length):
    """TODO guided implementation."""
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

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

### Source Mapping

- `dataset/lm_dataset.py:37-55`

### 常见坑

- 注意 max_length 必须至少容纳 BOS/EOS。
- labels 忽略 pad，input_ids 仍保留 pad。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？
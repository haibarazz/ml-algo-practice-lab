# KV cache 增量推理：每步只算新 token

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/06_inference/kv_cache_step_slice/kv_cache_step_slice.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


拆解 MiniMind generate 中 `input_ids[:, past_len:]` 和 attention 中 past K/V 拼接。

## 学习目标

- 理解 KV cache 为什么能降低自回归推理重复计算。
- 掌握 `past_len` 如何决定本步输入切片。
- 理解 K/V 拼接和 position offset 必须保持一致。

## MiniMind 源码定位

- `model/model_minimind.py:111-134`
- `model/model_minimind.py:209-232`
- `model/model_minimind.py:257-288`

## 源码机制详解

自回归生成第 t 步时，历史 token 的 K/V 已经在前面 step 算过。MiniMind 在 generate 中用 `past_len = past_key_values[0][0].shape[1]` 得到已缓存长度，再 forward `input_ids[:, past_len:]`，通常就是最新 token。
在 Attention 里，如果有 `past_key_value`，新算出的 `xk/xv` 会和历史 K/V 在 seq 维拼接。随后再 repeat_kv、转置 head 维，进入 attention 计算。
模型 forward 还用同一个 past length 作为 `start_pos`，从 RoPE cos/sin 表中切出正确位置。切片、KV 拼接和 RoPE offset 三者必须一致，否则生成会看错历史位置。

## 关键公式与数据流

- 无 cache 每步复杂度会重复处理整个前缀；有 cache 时每步只投影新 token 的 Q/K/V。
- $K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]$。
- $position_{new}$ 从 $past\_len$ 开始，而不是从 0 开始。

## 为什么练这个

- 这个练习把 generate 循环和 Attention.forward 连起来。
- 理解它后，KV cache 的 shape、显存和刷新问题会清晰很多。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def slice_new_tokens(input_ids, past_len):
    """带提示实现。"""
    # TODO 1: 输入是二维 list
    # TODO 2: 对每行保留 past_len 之后的 token
    # TODO 3: past_len 为 0 时返回完整输入
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def slice_new_tokens(input_ids, past_len):
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
def test_slice_new_tokens():
    ids = [[1, 5, 6], [1, 7, 8]]
    assert slice_new_tokens(ids, 0) == ids
    assert slice_new_tokens(ids, 2) == [[6], [8]]


test_slice_new_tokens()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def slice_new_tokens(input_ids, past_len):
    return [list(row[past_len:]) for row in input_ids]
```

### 解析

1. KV cache 不是缓存 logits，而是每层 attention 的 K/V。
2. attention_mask 仍要随生成长度增长。


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- 无 cache 每步复杂度会重复处理整个前缀；有 cache 时每步只投影新 token 的 Q/K/V。
- $K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]$。
- $position_{new}$ 从 $past\_len$ 开始，而不是从 0 开始。

### 易错点

- 切片错成 `input_ids[:, -1:]` 在某些多 token step 下会漏 token。
- position offset 从 0 重启会破坏 RoPE 相对位置。
- attention_mask 也要随新 token 扩展，否则 padding/可见性不一致。

### 面试追问

::: details 参考回答：KV cache 为什么只缓存 K/V，不缓存 Q？

Q 只属于当前 query 位置，每步新 token 都会产生新的 Q。历史 token 作为可被查询的记忆，需要保存的是它们的 K/V。

:::

::: details 参考回答：KV cache 的主要代价是什么？

代价是显存和带宽。长上下文下每层每个历史 token 都要保存 K/V，推理每步还要读取这些缓存，所以 GQA/MQA 会特别重要。

:::
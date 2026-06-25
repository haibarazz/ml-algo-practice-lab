# KV cache 增量推理：每步只算新 token笔记

## 关键公式与数据流

- 无 cache 每步复杂度会重复处理整个前缀；有 cache 时每步只投影新 token 的 Q/K/V。
- $K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]$。
- $position_{new}$ 从 $past\_len$ 开始，而不是从 0 开始。

## 易错点

- 切片错成 `input_ids[:, -1:]` 在某些多 token step 下会漏 token。
- position offset 从 0 重启会破坏 RoPE 相对位置。
- attention_mask 也要随新 token 扩展，否则 padding/可见性不一致。

## 面试追问

::: details 参考回答：KV cache 为什么只缓存 K/V，不缓存 Q？

Q 只属于当前 query 位置，每步新 token 都会产生新的 Q。历史 token 作为可被查询的记忆，需要保存的是它们的 K/V。

:::

::: details 参考回答：KV cache 的主要代价是什么？

代价是显存和带宽。长上下文下每层每个历史 token 都要保存 K/V，推理每步还要读取这些缓存，所以 GQA/MQA 会特别重要。

:::

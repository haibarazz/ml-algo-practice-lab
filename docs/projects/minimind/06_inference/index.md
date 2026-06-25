# 推理与采样

拆解 generate 中的 KV cache、temperature、top-k、top-p、重复惩罚和停止条件。

| 模块 | 学习重点 |
| --- | --- |
| [Top-k / Top-p 采样过滤：控制生成候选集合](./top_k_top_p_filter/top_k_top_p_filter.md) | 拆解 MiniMind `generate` 中 temperature、top-k、top-p 和 multinomial 的采样流程。 |
| [KV cache 增量推理：每步只算新 token](./kv_cache_step_slice/kv_cache_step_slice.md) | 拆解 MiniMind generate 中 `input_ids[:, past_len:]` 和 attention 中 past K/V 拼接。 |

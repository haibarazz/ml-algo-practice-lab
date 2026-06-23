# 推理

这个 section 按 MiniMind 学习路径组织，不按源码目录机械排序。

## Modules

- `top_k_top_p_filter`: 手写 MiniMind generate 中的采样过滤。
- `kv_cache_step_slice`: 理解 generate 中 past_key_values 对 input_ids 的切片。

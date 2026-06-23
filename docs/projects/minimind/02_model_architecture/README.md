# 模型结构

这个 section 按 MiniMind 学习路径组织，不按源码目录机械排序。

## Modules

- `rms_norm_minimind`: 手写 RMSNorm。
- `rope_rotate_half`: 复刻 MiniMind 的 rotate_half 形式 RoPE。
- `repeat_kv_for_gqa`: GQA/MQA 中把 KV heads 重复到 Q heads 数量。
- `swiglu_feed_forward`: 手写 MiniMind FFN 的门控结构。
- `moe_router_aux_loss`: MiniMind MoE 的 router 辅助损失。

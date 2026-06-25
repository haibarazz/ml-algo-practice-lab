# 模型结构

拆解 MiniMind Decoder-only Transformer 的归一化、位置编码、GQA、SwiGLU 与 MoE。

| 模块 | 学习重点 |
| --- | --- |
| [RMSNorm：只按均方根缩放的归一化](./rms_norm_minimind/rms_norm_minimind.md) | 拆解 MiniMind 中 `RMSNorm` 的公式、数值稳定项和它在 Transformer block 中的位置。 |
| [RoPE 旋转位置编码：MiniMind 的 rotate_half 写法](./rope_rotate_half/rope_rotate_half.md) | 从 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 理解 RoPE 怎样把位置信息注入 Q/K。 |
| [GQA/MQA：重复 KV head 以服务更多 Q head](./repeat_kv_for_gqa/repeat_kv_for_gqa.md) | 拆解 MiniMind 的 `repeat_kv` 和 attention 中 `num_key_value_heads` 的显存/带宽意义。 |
| [SwiGLU FFN：门控前馈网络](./swiglu_feed_forward/swiglu_feed_forward.md) | 拆解 MiniMind `FeedForward` 中 gate/up/down 三个投影如何组成现代 LLM 常见 FFN。 |
| [MoE 路由辅助损失：让专家负载更均衡](./moe_router_aux_loss/moe_router_aux_loss.md) | 拆解 MiniMind `MOEFeedForward` 的 top-k 路由、专家聚合和 aux loss。 |

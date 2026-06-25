# GQA/MQA：重复 KV head 以服务更多 Q head笔记

## 关键公式与数据流

- MHA：$H_q=H_k=H_v$；MQA：$H_k=H_v=1$；GQA：$1 < H_{kv} < H_q$。
- $n_{rep}=H_q/H_{kv}$。
- KV cache 规模近似正比于 $layers \times seq \times H_{kv}\times head\_dim$。

## 易错点

- 在 seq 维 repeat 会复制时间步，语义完全错。
- `num_attention_heads` 必须能被 `num_key_value_heads` 整除。
- 只看参数量会低估 GQA 价值，它主要优化推理 cache 和带宽。

## 面试追问

::: details 参考回答：GQA 相比 MHA 主要省在哪里？

主要省 KV cache 和推理时读取 K/V 的带宽。Q head 仍然多，但 K/V head 更少，历史 token 的缓存规模随 KV head 数下降。

:::

::: details 参考回答：`repeat_kv` 和真的学习更多 KV head 有什么区别？

`repeat_kv` 只是把少量已学习的 K/V 表示广播给多个 Q head，不增加 K/V 参数和 cache。学习更多 KV head 会增加表达多样性，但也增加显存和带宽成本。

:::

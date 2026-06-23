# GQA MQA Shapes Notes

## 核心公式

- MHA：$n_{qheads}=n_{kvheads}$；MQA：$n_{kvheads}=1$；GQA：$1<n_{kvheads}<n_{qheads}$。
- KV cache 规模近似正比于 $layers \times seq \times n_{kvheads} \times head\_dim$。

## 易错点

- 在 seq 维 repeat，而不是在 head 维 repeat K/V。
- 忘记 `target_heads` 必须是 `kv_heads` 的整数倍。
- repeat 后没有保持 head 顺序，导致 Q head 对错 KV group。
- 只看参数量而忽略推理时 KV cache 带宽和显存。

## 面试追问

- MHA、MQA、GQA 的 KV cache 大小差异是什么？
- 为什么大模型推理特别关注 KV cache？
- MQA 可能带来什么质量损失？GQA 如何折中？
- repeat_kv 和直接学习更多 KV head 的区别是什么？

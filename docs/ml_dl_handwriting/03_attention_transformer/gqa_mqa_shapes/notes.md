# GQA MQA Shapes Notes

## 易错点
- 在 seq 维 repeat 而不是 head 维。
- 忘记 target_heads 必须是 kv_heads 的整数倍。

## 面试追问
- MHA、MQA、GQA 的 KV cache 大小差异？
- 为什么大模型推理关注 KV cache？

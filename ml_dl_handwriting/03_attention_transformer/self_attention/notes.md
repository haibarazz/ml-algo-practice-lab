# Self Attention Notes

## 核心公式

- self-attention 中 $Q=XW_Q$，$K=XW_K$，$V=XW_V$，三者来自同一序列。
- 输出 $O=Attention(Q,K,V)$，长度通常等于 query 序列长度。

## 易错点

- Q/K/V shape 不一致，尤其是最后一维和 head 维。
- 把 self-attention 和 cross-attention 混淆；self-attention 的 Q/K/V 来自同一输入。
- 没有保留 batch 维，单样本测试通过但 batch 输入失败。
- 忘记 mask 时，decoder 会看到未来 token。

## 面试追问

- self-attention 的时间复杂度和序列长度是什么关系？
- 为什么 Transformer 用 self-attention 可以并行处理序列？
- self-attention 如何捕获长距离依赖？
- 为什么需要多头 attention，而不是只用一个头？

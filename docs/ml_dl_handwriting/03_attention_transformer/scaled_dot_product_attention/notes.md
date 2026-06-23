# Scaled Dot Product Attention Notes

## 核心公式

- $Attention(Q,K,V)=softmax(\frac{QK^\top}{\sqrt{d_k}})V$。
- mask 通常在 softmax 前加到 logits 上，被屏蔽位置加一个很大的负数。

## 易错点

- 忘记除以 $\sqrt{d_k}$，维度大时 logits 方差过大，softmax 容易饱和。
- mask 在 softmax 后处理会导致概率和不为 1。
- K 的转置维度写错，batch/head/seq 维容易混乱。
- 全 mask 行可能产生 `nan`，工程上要定义处理策略。

## 面试追问

- 为什么 dot-product attention 需要 scaling？
- causal mask 和 padding mask 分别解决什么问题？
- attention 的时间和显存复杂度是多少？
- 为什么 softmax 前加大负数可以实现 mask？

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

::: details 参考回答：为什么 dot-product attention 需要 scaling？

如果 Q 和 K 的各维独立且方差相近，点积的方差会随 `d_k` 增大而增大。除以 `sqrt(d_k)` 可以稳定 logits 尺度，避免 softmax 太早饱和导致梯度变小。

:::

::: details 参考回答：causal mask 和 padding mask 分别解决什么问题？

causal mask 防止当前位置看到未来 token，是自回归生成的因果约束。padding mask 屏蔽补齐位置，避免模型把 pad token 当作真实上下文参与注意力。

:::

::: details 参考回答：attention 的时间和显存复杂度是多少？

标准 attention 的 score 矩阵大小是 `seq_len^2`，所以时间复杂度和显存复杂度都近似 `O(n^2)`。多头和 batch 会再乘上对应维度，因此长上下文下瓶颈非常明显。

:::

::: details 参考回答：为什么 softmax 前加大负数可以实现 mask？

softmax 前加一个很大的负数，会让对应位置的指数接近 0，从而概率接近 0。这样剩余未屏蔽位置仍能正常归一化，概率和保持为 1。

:::

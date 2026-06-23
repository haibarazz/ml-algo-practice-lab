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

::: details 参考回答：self-attention 的时间复杂度和序列长度是什么关系？

self-attention 需要计算序列内任意 token 两两交互，score 矩阵是 `n x n`。因此时间和显存随序列长度大致二次增长，这是长上下文 Transformer 的核心瓶颈。

:::

::: details 参考回答：为什么 Transformer 用 self-attention 可以并行处理序列？

self-attention 不像 RNN 那样必须按时间步递推，Q/K/V 可以对所有位置一次性投影。只要 mask 处理好，训练时每个位置的注意力计算可以并行完成。

:::

::: details 参考回答：self-attention 如何捕获长距离依赖？

每个 token 可以直接对任意距离的 token 分配注意力权重，所以路径长度是常数级的。相比 RNN 需要逐步传播信息，self-attention 更容易建模远距离依赖。

:::

::: details 参考回答：为什么需要多头 attention，而不是只用一个头？

多头让不同子空间学习不同关系，比如局部、长程、语法或位置模式。单头即使维度很大，也只有一套注意力分布，多头提供了多组并行的对齐方式。

:::

# MHA With Mask Notes

## 核心公式

- $scores=QK^\top/\sqrt{d_k}+mask$，其中被屏蔽位置通常为 $-\infty$。
- $weights=softmax(scores)$，$O=weightsV$。

## 易错点

- mask 语义反了：True 是保留还是屏蔽必须在实现中固定。
- 在 softmax 后 mask，概率和不再为 1。
- mask shape 没有正确广播到 batch/head/query/key 维。
- padding mask 和 causal mask 叠加时 dtype、shape 容易错。

## 面试追问

::: details 参考回答：causal mask 和 padding mask 的区别是什么？

causal mask 屏蔽未来位置，用于保证自回归生成不能偷看答案。padding mask 屏蔽补齐 token，用于避免变长 batch 中的无效位置参与注意力。

:::

::: details 参考回答：训练和自回归推理时 mask 有什么不同？

训练时通常一次性给完整序列加 causal mask，并行计算所有位置的 loss。自回归推理时每步只生成一个新 token，并通过 KV cache 复用历史 K/V，mask 逻辑更多体现为 position offset 和可见历史。

:::

::: details 参考回答：为什么 mask 应该作用在 softmax 前？

mask 放在 softmax 前，屏蔽位置的概率会在归一化时变成接近 0，同时剩余位置概率和仍为 1。若 softmax 后再 mask，会破坏概率分布，并改变输出尺度。

:::

::: details 参考回答：如果一整行都被 mask，softmax 会出现什么问题？

一整行都被 mask 时，softmax 的输入可能全是负无穷，结果会产生 NaN。工程上需要避免这种样本，或对全 mask 行做特殊处理，比如输出全 0 并跳过对应 loss。

:::

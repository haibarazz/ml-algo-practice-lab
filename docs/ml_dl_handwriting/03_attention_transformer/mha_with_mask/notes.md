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

- causal mask 和 padding mask 的区别是什么？
- 训练和自回归推理时 mask 有什么不同？
- 为什么 mask 应该作用在 softmax 前？
- 如果一整行都被 mask，softmax 会出现什么问题？

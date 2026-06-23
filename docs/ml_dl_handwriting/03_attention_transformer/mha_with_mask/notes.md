# MHA With Mask Notes

## 易错点
- mask 语义反了：True 是保留还是屏蔽必须说明。
- 在 softmax 后 mask 导致概率和不为 1。

## 面试追问
- causal mask 和 padding mask 区别？
- 训练和推理时 mask 有什么不同？

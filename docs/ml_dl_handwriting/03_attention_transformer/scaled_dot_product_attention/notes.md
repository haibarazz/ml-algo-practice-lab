# Scaled Dot Product Attention Notes

## 易错点
- 忘记除以 sqrt(d)。
- mask 在 softmax 后处理。
- K 的转置维度写错。

## 面试追问
- 为什么 attention 需要 scaling？
- causal mask 怎么构造？

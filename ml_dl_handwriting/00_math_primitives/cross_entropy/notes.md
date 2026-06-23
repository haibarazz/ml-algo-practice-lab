# Cross Entropy Notes

## 易错点
- 先 softmax 再 log 容易数值不稳定。
- target 下标和 one-hot 表示混用。
- 忘记 batch mean。

## 面试追问
- softmax + cross entropy 的梯度为什么是 `p - y`？
- 为什么框架里常用 fused cross entropy？

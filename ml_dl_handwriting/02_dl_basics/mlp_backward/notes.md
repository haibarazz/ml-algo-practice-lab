# MLP Backward Notes

## 易错点

- MSE 梯度分母用 batch size 还是元素总数要和 loss 定义一致。
- ReLU backward 用 `a1 > 0` 和 `z1 > 0` 在普通 ReLU 下等价，但推荐缓存 `z1`。
- `db` 忘记沿 batch 维求和。
- 矩阵乘法方向写反，导致 shape 对不上。

## 面试追问

- 如果输出是多分类 softmax + cross entropy，最后一层梯度怎么化简？
- 为什么需要激活函数？
- 梯度消失和梯度爆炸分别可能出现在哪里？
- 如何用数值梯度检查 backward 是否正确？

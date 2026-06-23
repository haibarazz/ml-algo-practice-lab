# MLP Backward Notes

## 核心公式

- 链式法则：$\frac{\partial L}{\partial W_2}=h^\top\frac{\partial L}{\partial \hat y}$，$\frac{\partial L}{\partial h}=\frac{\partial L}{\partial \hat y}W_2^\top$。
- ReLU backward：$\frac{\partial h}{\partial z_1}=\mathbf{1}[z_1>0]$。

## 易错点

- MSE 梯度分母用 batch size 还是元素总数，要和 loss 定义一致。
- ReLU backward 推荐缓存 `z1`，不要依赖被后续修改过的激活值。
- `db` 忘记沿 batch 维求和。
- 矩阵乘法方向写反，导致 shape 对不上或梯度转置。

## 面试追问

- 如果输出层改成 softmax + cross entropy，最后一层梯度如何化简？
- 为什么需要激活函数？没有激活时 backward 会发生什么？
- 梯度消失和梯度爆炸分别可能出现在哪里？
- 如何用数值梯度检查 backward 是否正确？

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

::: details 参考回答：如果输出层改成 softmax + cross entropy，最后一层梯度如何化简？

softmax + cross entropy 时，最后一层 logits 的梯度可以直接写成 `probs - onehot`，再按 batch 或 token 平均。这样不用显式写 softmax 的完整雅可比矩阵，数值和实现都更稳定。

:::

::: details 参考回答：为什么需要激活函数？没有激活时 backward 会发生什么？

没有激活函数时，整个网络只是线性层的复合，forward 等价于一个线性模型。backward 仍然能算，但训练再深也不会增加非线性表达能力，只会让参数冗余和优化更复杂。

:::

::: details 参考回答：梯度消失和梯度爆炸分别可能出现在哪里？

梯度消失常出现在深层链式乘法中，尤其是 sigmoid/tanh 饱和区或不合适初始化下。梯度爆炸也来自多层雅可比连乘，RNN、深层网络或学习率过大时更明显。

:::

::: details 参考回答：如何用数值梯度检查 backward 是否正确？

数值梯度检查用有限差分近似某个参数的导数，再和解析梯度比较相对误差。检查时要用小网络、固定随机种子、关闭 dropout，并选择合适 epsilon，避免浮点误差主导。

:::

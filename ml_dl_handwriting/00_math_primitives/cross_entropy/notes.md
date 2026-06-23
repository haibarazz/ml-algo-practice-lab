# Cross Entropy Notes

## 核心公式

- one-hot target 下：$L=-\sum_c y_c\log p_c$；若 target 是类别下标 $t$，则 $L=-\log p_t$。
- 对 logits 使用 softmax + CE 时，常见梯度为 $\frac{\partial L}{\partial z}=p-y$。

## 易错点

- 先 softmax 再 log 容易数值不稳定，应使用 log-sum-exp 或 fused cross entropy。
- target 下标和 one-hot 表示混用，导致 shape 或语义错误。
- 忘记 batch mean，使 loss 尺度随 batch size 变化。
- 把多分类单标签 CE 和多标签 BCE 混淆。

## 面试追问

- softmax + cross entropy 的梯度为什么可以化简成 `p - y`？
- 为什么 PyTorch 的 `CrossEntropyLoss` 输入通常是 logits 而不是概率？
- label smoothing 会怎样改变 target 分布和梯度？
- 类别不均衡时，cross entropy 可以怎样加权？

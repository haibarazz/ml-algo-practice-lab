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

::: details 参考回答：softmax + cross entropy 的梯度为什么可以化简成 `p - y`？

对 logits 求导时，softmax 的雅可比矩阵和 `-log p_y` 的梯度会合并，最终得到 `p - y`。这个结果直观上表示预测概率比目标概率高的类别要被压低，目标类别要被抬高。

:::

::: details 参考回答：为什么 PyTorch 的 `CrossEntropyLoss` 输入通常是 logits 而不是概率？

logits 可以进入 fused cross entropy，用 log-sum-exp 做稳定计算，避免先 softmax 后 log 的数值问题。如果输入已经是概率，极小概率会导致 `log(0)` 或梯度不稳定，而且还可能重复 softmax。

:::

::: details 参考回答：label smoothing 会怎样改变 target 分布和梯度？

label smoothing 会把 one-hot 目标改成“目标类接近 1、非目标类有少量概率”的软标签。这样梯度不再把正确类概率强推到 1，可以减轻过度自信，常对泛化和校准有帮助。

:::

::: details 参考回答：类别不均衡时，cross entropy 可以怎样加权？

可以给不同类别设置 class weight，让少数类样本的 loss 权重更大。工程上也常结合重采样、focal loss 或按业务成本设置权重，但要注意验证集指标是否真的改善。

:::

# MSE Loss Notes

## 核心公式

- $MSE=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$。
- 若对预测值求梯度：$\frac{\partial MSE}{\partial \hat y_i}=\frac{2}{N}(\hat y_i-y_i)$。

## 易错点

- sum 和 mean 混用会导致 loss 与梯度尺度不一致。
- `y_true - y_pred` 和 `y_pred - y_true` 对 loss 无影响，但对梯度方向有影响。
- 多维输出时要说清楚是按所有元素平均，还是先按样本再按输出维聚合。
- MSE 对异常值敏感，数据噪声重尾时不一定是好选择。

## 面试追问

::: details 参考回答：MSE 为什么比 MAE 更容易受异常值影响？

MSE 对误差做平方，误差变大时惩罚是二次增长；MAE 是线性增长。异常值往往误差很大，所以在 MSE 下会主导 loss 和梯度。

:::

::: details 参考回答：MSE、RMSE、MAE 分别适合什么回归评价场景？

MSE 适合噪声接近高斯、希望大误差被更强惩罚的回归训练。RMSE 和原标签同量纲，便于解释；MAE 对异常值更稳健，适合重尾噪声或更关心绝对偏差的场景。

:::

::: details 参考回答：如果训练时用 MSE，梯度分母应该除以 batch size 还是元素总数？

分母要和 loss 的定义一致：如果 loss 是所有元素的 mean，梯度就除以元素总数；如果先按样本聚合再平均，就除以 batch size。面试里关键不是固定答案，而是 loss 数值和梯度尺度必须一致。

:::

::: details 参考回答：Huber loss 如何在 MSE 和 MAE 之间折中？

Huber loss 在小误差区间使用平方惩罚，保持 MSE 的平滑梯度；在大误差区间切换成近似线性惩罚，降低异常值影响。阈值 delta 控制从 MSE 到 MAE 的切换点。

:::

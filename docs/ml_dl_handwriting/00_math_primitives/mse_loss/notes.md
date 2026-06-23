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

- MSE 为什么比 MAE 更容易受异常值影响？
- MSE、RMSE、MAE 分别适合什么回归评价场景？
- 如果训练时用 MSE，梯度分母应该除以 batch size 还是元素总数？
- Huber loss 如何在 MSE 和 MAE 之间折中？

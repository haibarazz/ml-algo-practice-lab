# MSE Loss Notes

## 易错点
- sum 和 mean 混用导致梯度尺度不一致。
- `y_true - y_pred` 和 `y_pred - y_true` 对 loss 无影响，但对梯度方向有影响。

## 面试追问
- MSE 对异常值敏感吗？
- MAE 和 MSE 的优化差异是什么？

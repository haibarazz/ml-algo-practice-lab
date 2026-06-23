# SVM Hinge Loss Notes

## 核心公式

- 二分类 hinge loss：$L=\max(0,1-y(w^\top x+b))$，其中 $y\in\{-1,+1\}$。
- 带 L2 正则的软间隔目标常写为 $\frac{1}{2}\lVert w\rVert^2+C\sum_i\xi_i$。

## 易错点

- 标签用 0/1 而不是 -1/+1，会让 margin 公式失效。
- 忘记正则项，无法体现最大间隔。
- active mask 条件写反；只有 margin 小于 1 的样本贡献 hinge 梯度。
- hinge loss 在折点不可导，通常使用次梯度。

## 面试追问

- SVM 的 margin 几何意义是什么？
- hinge loss 和 logistic loss 的差别是什么？
- C 参数变大或变小会怎样影响间隔和误分类？
- 核 SVM 为什么在大样本上训练代价高？

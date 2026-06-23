# KMeans One Step Notes

## 核心公式

- assignment：$z_i=\arg\min_k\lVert x_i-\mu_k\rVert^2$。
- update：$\mu_k=\frac{1}{|C_k|}\sum_{i:z_i=k}x_i$。

## 易错点

- 距离矩阵 shape 写反，导致样本维和中心维混淆。
- assignment 只需要比较平方距离，不必开根号。
- 空簇需要明确策略：保持旧中心、重采样或选择最远点。
- `np.argmin` 默认返回第一个最小值，tie-break 要说明。

## 面试追问

- KMeans 一轮迭代为什么分为 assignment 和 update？
- KMeans 对初始中心敏感吗？KMeans++ 解决了什么问题？
- K 怎么选？肘部法和 silhouette score 的直觉是什么？
- 如果数据量很大，如何加速 assignment 或做 mini-batch KMeans？

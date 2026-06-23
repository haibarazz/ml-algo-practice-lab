# KMeans One Step Notes

## 易错点

- 距离矩阵 shape 写反。
- 用真实欧氏距离开根号，虽然结果相同但没必要。
- 忘记空簇处理。
- tie-break 没说明。`np.argmin` 默认返回第一个最小值。

## 面试追问

- KMeans 对初始中心敏感吗？
- K 怎么选？
- KMeans 和 DBSCAN 的区别？
- 如果数据量很大，如何加速 assignment？

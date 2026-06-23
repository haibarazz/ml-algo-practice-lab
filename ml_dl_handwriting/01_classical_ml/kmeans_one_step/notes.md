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

::: details 参考回答：KMeans 一轮迭代为什么分为 assignment 和 update？

assignment 固定中心，把每个样本分到最近的簇；update 固定分配，用簇内均值更新中心。两步交替分别优化同一个平方误差目标中的不同变量。

:::

::: details 参考回答：KMeans 对初始中心敏感吗？KMeans++ 解决了什么问题？

KMeans 对初始中心很敏感，因为目标函数非凸，初始化不同可能落到不同局部最优。KMeans++ 倾向选择彼此较远的初始中心，降低坏初始化概率，加快收敛。

:::

::: details 参考回答：K 怎么选？肘部法和 silhouette score 的直觉是什么？

肘部法看 inertia 随 K 增加的下降曲线，在收益明显变小的拐点选 K。silhouette score 同时看簇内紧密和簇间分离，越高通常表示聚类结构越清晰。

:::

::: details 参考回答：如果数据量很大，如何加速 assignment 或做 mini-batch KMeans？

可以用矩阵化距离、分块计算、近似最近中心搜索来加速 assignment。数据很大时常用 mini-batch KMeans，每次只用小批样本更新中心，用更低成本逼近全量结果。

:::

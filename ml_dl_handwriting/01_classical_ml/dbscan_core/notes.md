# DBSCAN Core Notes

## 核心公式

- 核心点：$|N_\varepsilon(x)|\ge minPts$，其中 $N_\varepsilon(x)=\{y:\operatorname{dist}(x,y)\le\varepsilon\}$。
- 密度可达由核心点邻域扩展得到，无法从任何核心点扩展到的点可标为噪声。

## 易错点

- `min_samples` 是否包含自身要说明，本项目约定包含自身。
- 噪声点后续可能变成边界点，不能过早永久丢弃。
- `eps` 对尺度敏感，输入特征通常需要标准化。
- 不同密度簇上 DBSCAN 可能表现不好，一个全局 eps 难以兼顾。

## 面试追问

::: details 参考回答：DBSCAN 和 KMeans 的核心区别是什么？

KMeans 假设用 K 个中心解释数据，需要预设簇数，并倾向球状簇。DBSCAN 基于密度连通，不需要预设 K，可以识别噪声和非凸形状簇。

:::

::: details 参考回答：DBSCAN 为什么可以发现任意形状簇并识别噪声？

只要核心点之间能通过 eps 邻域链式连接，DBSCAN 就能把它们扩展成同一簇，所以不要求簇是球形。低密度区域无法连接到任何核心点，就会被标记为噪声或边界外样本。

:::

::: details 参考回答：eps 和 min_samples 应该如何选择？

eps 通常结合距离分布或 k-distance 曲线选择，寻找从密集到稀疏的拐点。min_samples 可根据维度和噪声水平设定，维度越高或噪声越多通常需要更谨慎调参。

:::

::: details 参考回答：DBSCAN 为什么不天然支持对新样本直接 predict？

DBSCAN 的簇定义依赖训练集中的密度连通结构，而不是学习一个显式参数化边界。新样本来了以后，它可能改变局部密度关系，所以原始算法没有像 KMeans 那样天然的 `predict`。

:::

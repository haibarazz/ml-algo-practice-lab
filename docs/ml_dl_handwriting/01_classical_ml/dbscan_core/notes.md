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

- DBSCAN 和 KMeans 的核心区别是什么？
- DBSCAN 为什么可以发现任意形状簇并识别噪声？
- eps 和 min_samples 应该如何选择？
- DBSCAN 为什么不天然支持对新样本直接 predict？

# EM GMM One Step Notes

## 核心公式

- E-step：$\gamma_{ik}=\frac{\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}$。
- M-step：$N_k=\sum_i\gamma_{ik}$，$\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i$，$\pi_k=N_k/n$。

## 易错点

- E-step 应按每个样本在所有簇上的责任度归一化。
- M-step 更新方差时要使用新均值，而不是旧均值。
- 方差过小会导致数值不稳定，工程中常加最小方差或协方差正则。
- 概率密度连乘容易下溢，高维实现常用 log-sum-exp。

## 面试追问

::: details 参考回答：EM 为什么能单调不降低数据似然？

EM 通过构造当前参数下的后验责任度，优化似然的一个下界。E-step 固定参数更新下界，M-step 最大化下界，因此每轮不会降低原始数据似然。

:::

::: details 参考回答：GMM 和 KMeans 的关系是什么？什么时候 GMM 会退化得像 KMeans？

GMM 是软聚类，每个样本对每个高斯都有责任度；KMeans 是硬分配，只属于最近中心。若协方差相同且趋近很小，GMM 的责任度会接近 one-hot，表现就很像 KMeans。

:::

::: details 参考回答：GMM 中协方差矩阵 full、diag、spherical 有什么差别？

full covariance 可以表达任意椭圆形簇，但参数多、计算贵；diag 只建模各维独立方差，成本较低；spherical 每个簇只有一个方差，更接近球状簇假设。

:::

::: details 参考回答：EM 对初始化敏感吗？如何缓解局部最优？

EM 对初始化敏感，因为似然目标非凸，可能收敛到局部最优或奇异解。常见缓解方式是多次随机初始化、用 KMeans 初始化均值、给协方差加正则和设置最小方差。

:::

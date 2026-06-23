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

- EM 为什么能单调不降低数据似然？
- GMM 和 KMeans 的关系是什么？什么时候 GMM 会退化得像 KMeans？
- GMM 中协方差矩阵 full、diag、spherical 有什么差别？
- EM 对初始化敏感吗？如何缓解局部最优？

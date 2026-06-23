# EM GMM One Step

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

GMM 的 EM 一轮：

E-step 计算责任度：

$$r_{ik}=\frac{\pi_k N(x_i|\mu_k,\sigma_k^2)}{\sum_j \pi_j N(x_i|\mu_j,\sigma_j^2)}$$

M-step 用责任度加权更新权重、均值和方差。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def em_gmm_one_step(x, weights, means, variances):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np

def em_gmm_one_step(x, weights, means, variances):
    """TODO blank implementation."""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现无提示函数后直接运行测试区代码。

```python
def test_em_gmm_one_step():
    x = np.array([-1.0, 0.0, 5.0, 6.0])
    resp, w, m, v = em_gmm_one_step(x, np.array([0.5, 0.5]), np.array([0.0, 5.0]), np.array([1.0, 1.0]))
    assert resp.shape == (4, 2)
    assert np.allclose(resp.sum(axis=1), np.ones(4))
    assert m[0] < m[1]
    assert np.allclose(w.sum(), 1.0)

test_em_gmm_one_step()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _normal_pdf(x, mean, var):
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

def em_gmm_one_step(x, weights, means, variances):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    probs = np.stack([weights[k] * _normal_pdf(x, means[k], variances[k]) for k in range(len(weights))], axis=1)
    resp = probs / probs.sum(axis=1, keepdims=True)
    Nk = resp.sum(axis=0)
    new_weights = Nk / len(x)
    new_means = (resp * x[:, None]).sum(axis=0) / Nk
    new_vars = (resp * (x[:, None] - new_means[None, :]) ** 2).sum(axis=0) / Nk
    return resp, new_weights, new_means, new_vars
```

### 解析

1. E-step 先算每个分量的加权密度。
2. 责任度按每个样本归一化。
3. M-step 用责任度作为软计数更新参数。
4. 方差更新要围绕新均值。

## 工程要点 / 面试追问

### 核心公式

- E-step：$\gamma_{ik}=\frac{\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}$。
- M-step：$N_k=\sum_i\gamma_{ik}$，$\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i$，$\pi_k=N_k/n$。

### 易错点

- E-step 应按每个样本在所有簇上的责任度归一化。
- M-step 更新方差时要使用新均值，而不是旧均值。
- 方差过小会导致数值不稳定，工程中常加最小方差或协方差正则。
- 概率密度连乘容易下溢，高维实现常用 log-sum-exp。

### 面试追问

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

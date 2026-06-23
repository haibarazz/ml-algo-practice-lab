# EM GMM One Step Solution

## Reference Implementation

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

## Explanation

1. E-step 先算每个分量的加权密度。
2. 责任度按每个样本归一化。
3. M-step 用责任度作为软计数更新参数。
4. 方差更新要围绕新均值。

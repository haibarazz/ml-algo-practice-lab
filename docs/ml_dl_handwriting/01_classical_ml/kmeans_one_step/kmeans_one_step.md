# KMeans One Step

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/kmeans_one_step/kmeans_one_step.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: sample complete

## 手写实现约束

允许使用 NumPy；不允许调用 sklearn.cluster.KMeans 或任何现成聚类实现。

## 原理最小说明

KMeans 每轮包含两步：

1. Assignment: 对每个样本，找到距离最近的聚类中心。
2. Update: 对每个簇，取被分到该簇的样本均值作为新中心。

核心距离通常使用欧氏距离平方：

$$
\|x-c\|_2^2 = \sum_d (x_d-c_d)^2
$$

本模块只实现一轮迭代，重点是两两距离矩阵、最近中心索引和中心更新。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def kmeans_one_step_guided(points, centers):
    """Run one KMeans assignment-update step."""
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    # TODO 1: compute squared distances with shape [num_points, num_centers]
    # diff = ?
    # distances = ?

    # TODO 2: assign each point to nearest center
    # labels = ?

    # TODO 3: update each center by averaging assigned points
    # If a cluster receives no point, keep its old center.
    # new_centers = ?

    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def kmeans_one_step(points, centers):
    """Run one KMeans assignment-update step.

    Returns:
        labels: shape [num_points]
        new_centers: shape [num_centers, dim]
    """
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现 `blank` 函数后直接运行：

```python
def test_kmeans_one_step():
    points = np.array([
        [0.0, 0.0],
        [0.0, 2.0],
        [10.0, 10.0],
        [10.0, 12.0],
    ])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, new_centers = kmeans_one_step(points, centers)
    assert labels.tolist() == [0, 0, 1, 1]
    assert np.allclose(new_centers, np.array([[0.0, 1.0], [10.0, 11.0]]))

    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    centers = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, new_centers = kmeans_one_step(points, centers)
    assert labels.tolist() == [0, 0]
    assert np.allclose(new_centers[0], np.array([0.5, 0.0]))
    assert np.allclose(new_centers[1], centers[1])  # empty cluster keeps old center


test_kmeans_one_step()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np


def kmeans_one_step(points, centers):
    """Run one KMeans assignment-update step.

    Returns:
        labels: shape [num_points]
        new_centers: shape [num_centers, dim]
    """
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    diff = points[:, None, :] - centers[None, :, :]
    distances = np.sum(diff * diff, axis=-1)
    labels = np.argmin(distances, axis=1)

    new_centers = centers.copy()
    for k in range(centers.shape[0]):
        assigned = points[labels == k]
        if assigned.shape[0] > 0:
            new_centers[k] = assigned.mean(axis=0)

    return labels, new_centers
```

### 解析

1. `points[:, None, :] - centers[None, :, :]` 得到 `[N, K, D]` 的广播差值。
2. 沿最后一维求平方和，得到 `[N, K]` 距离矩阵。
3. `argmin(axis=1)` 给每个样本选最近中心。
4. 更新中心时必须处理空簇。这个样板选择保留旧中心，是面试里最容易漏掉的边界。


:::

## 工程要点 / 面试追问

### 核心公式

- assignment：$z_i=\arg\min_k\lVert x_i-\mu_k\rVert^2$。
- update：$\mu_k=\frac{1}{|C_k|}\sum_{i:z_i=k}x_i$。

### 易错点

- 距离矩阵 shape 写反，导致样本维和中心维混淆。
- assignment 只需要比较平方距离，不必开根号。
- 空簇需要明确策略：保持旧中心、重采样或选择最远点。
- `np.argmin` 默认返回第一个最小值，tie-break 要说明。

### 面试追问

- KMeans 一轮迭代为什么分为 assignment 和 update？
- KMeans 对初始中心敏感吗？KMeans++ 解决了什么问题？
- K 怎么选？肘部法和 silhouette score 的直觉是什么？
- 如果数据量很大，如何加速 assignment 或做 mini-batch KMeans？
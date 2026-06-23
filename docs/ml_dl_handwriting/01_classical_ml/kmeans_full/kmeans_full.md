# KMeans Full

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/kmeans_full/kmeans_full.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: 实现 KMeans 多轮迭代。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

完整 KMeans 重复执行 assignment 和 update，直到中心变化小于阈值或达到最大迭代次数。

本模块沿用空簇保留旧中心的策略。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def kmeans_full(points, initial_centers, max_iters=100, tol=1e-6):
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

def kmeans_full(points, initial_centers, max_iters=100, tol=1e-6):
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
def test_kmeans_full():
    points = np.array([[0, 0], [0, 2], [10, 10], [10, 12]], dtype=float)
    labels, centers = kmeans_full(points, np.array([[0, 0], [10, 10]], dtype=float))
    assert labels.tolist() == [0, 0, 1, 1]
    assert np.allclose(centers, np.array([[0, 1], [10, 11]], dtype=float))

test_kmeans_full()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def kmeans_full(points, initial_centers, max_iters=100, tol=1e-6):
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(initial_centers, dtype=np.float64).copy()
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for k in range(centers.shape[0]):
            assigned = points[labels == k]
            if len(assigned) > 0:
                new_centers[k] = assigned.mean(axis=0)
        if np.linalg.norm(new_centers - centers) < tol:
            centers = new_centers
            break
        centers = new_centers
    return labels, centers
```

### 解析

1. 每轮先重新分配标签。
2. 再用每个簇的样本均值更新中心。
3. 中心变化小于 `tol` 即停止。
4. 空簇保留旧中心，避免 NaN。


:::

## 工程要点 / 面试追问

见 `notes.md`。
# DBSCAN Core

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/dbscan_core/dbscan_core.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 题源线索

- Topic: DBSCAN 核心点与密度可达。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

DBSCAN 用两个参数定义簇：`eps` 邻域半径和 `min_samples` 最小邻居数。

核心点：eps 邻域内样本数不少于 `min_samples` 的点。

本模块实现简单 DBSCAN，返回每个点的簇标签，噪声为 -1。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def dbscan_core(points, eps, min_samples):
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

def dbscan_core(points, eps, min_samples):
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
def test_dbscan_core():
    points = np.array([[0, 0], [0, 0.1], [5, 5], [5.1, 5], [10, 10]], dtype=float)
    labels = dbscan_core(points, eps=0.25, min_samples=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] == -1

test_dbscan_core()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def dbscan_core(points, eps, min_samples):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(distances[i] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.where(distances[j] <= eps)[0].tolist()
                if len(j_neighbors) >= min_samples:
                    for nb in j_neighbors:
                        if nb not in queue:
                            queue.append(nb)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels
```

### 解析

1. 先预计算两两距离矩阵，便于教学。
2. 遍历未访问点，判断是否核心点。
3. 核心点启动 BFS 扩展簇。
4. 噪声点可能后来被某个簇吸收。


:::

## 工程要点 / 面试追问

见 `notes.md`。
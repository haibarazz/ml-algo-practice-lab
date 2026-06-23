# DBSCAN Core

> Status: complete

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

## 工程要点 / 面试追问

### 核心公式

- 核心点：$|N_\varepsilon(x)|\ge minPts$，其中 $N_\varepsilon(x)=\{y:\operatorname{dist}(x,y)\le\varepsilon\}$。
- 密度可达由核心点邻域扩展得到，无法从任何核心点扩展到的点可标为噪声。

### 易错点

- `min_samples` 是否包含自身要说明，本项目约定包含自身。
- 噪声点后续可能变成边界点，不能过早永久丢弃。
- `eps` 对尺度敏感，输入特征通常需要标准化。
- 不同密度簇上 DBSCAN 可能表现不好，一个全局 eps 难以兼顾。

### 面试追问

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

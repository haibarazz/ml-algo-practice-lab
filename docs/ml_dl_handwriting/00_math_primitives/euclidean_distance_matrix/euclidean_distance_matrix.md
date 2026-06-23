# Euclidean Distance Matrix

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/00_math_primitives/euclidean_distance_matrix/euclidean_distance_matrix.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

给定 $X \in R^{m\times d}$ 和 $Y \in R^{n\times d}$，距离矩阵 $D \in R^{m\times n}$ 的元素为：

$$D_{ij}=\sqrt{\sum_k (X_{ik}-Y_{jk})^2}$$

实现重点是利用广播得到 `[m, n, d]` 的差值张量，然后沿最后一维求和。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def euclidean_distance_matrix(X, Y):
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

def euclidean_distance_matrix(X, Y):
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
def test_euclidean_distance_matrix():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    Y = np.array([[0.0, 0.0], [6.0, 8.0]])
    D = euclidean_distance_matrix(X, Y)
    assert D.shape == (2, 2)
    assert np.allclose(D, np.array([[0.0, 10.0], [5.0, 5.0]]))
    assert np.allclose(euclidean_distance_matrix(X, X).diagonal(), np.zeros(2))

test_euclidean_distance_matrix()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def euclidean_distance_matrix(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))
```

### 解析

1. `X[:, None, :]` 把 X 扩成 `[m, 1, d]`。
2. `Y[None, :, :]` 把 Y 扩成 `[1, n, d]`。
3. 广播相减后沿特征维求平方和，再开方。
4. 实际 KNN / KMeans 中常用平方距离避免开方。


:::

## 工程要点 / 面试追问

### 核心公式

- $D_{ij}=\lVert x_i-y_j\rVert_2$；实现时常先算平方距离：$\lVert x_i\rVert^2+\lVert y_j\rVert^2-2x_i^\top y_j$。
- 如果只比较最近邻，平方距离和真实距离排序一致，可以省掉 $\sqrt{\cdot}$。

### 易错点

- 把输出 shape 写反；给定 X shape 为 `[m,d]`、Y shape 为 `[n,d]` 时，结果应为 `[m,n]`。
- 显式构造 `[m,n,d]` 中间张量会占用大量内存，大数据时优先使用矩阵乘法展开式。
- 浮点误差可能让平方距离出现极小负数，开方前可做 `clip(min=0)`。
- 特征尺度差异会直接影响欧氏距离，KNN/KMeans 前通常要考虑标准化。

### 面试追问

::: details 参考回答：如何用矩阵乘法实现两两欧氏距离？时间复杂度和内存复杂度分别是多少？

可以利用恒等式 `||x-y||^2 = ||x||^2 + ||y||^2 - 2xy`，先分别算出 X 和 Y 的平方范数，再用矩阵乘法得到两两内积。时间复杂度仍是 `O(mnd)`，但避免显式构造 `[m,n,d]` 的差分张量，中间内存主要是输出矩阵 `O(mn)`。

:::

::: details 参考回答：为什么最近邻检索时常比较平方距离而不是距离本身？

平方根是单调递增函数，所以比较距离大小时，平方距离和真实欧氏距离的排序完全一致。最近邻只关心谁更小，不关心距离的物理单位，因此省掉开方可以减少计算和数值误差。

:::

::: details 参考回答：余弦相似度和欧氏距离分别适合什么特征表示？

欧氏距离适合特征已经对齐、尺度有意义、向量范数也携带信息的场景，比如标准化后的连续特征。余弦相似度更关注方向，适合文本 embedding、语义向量这类范数容易受长度或置信度影响的表示。

:::

::: details 参考回答：如果样本量很大，如何分块计算距离矩阵以避免内存爆掉？

可以按 query 样本分块，逐块计算 `[block_size, n]` 的距离，再把结果写入输出或只保留 top-k。block size 由可用显存或内存决定，核心原则是不要同时物化完整 `[m,n,d]` 或过大的 `[m,n]` 临时矩阵。

:::
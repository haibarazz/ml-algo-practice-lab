# Euclidean Distance Matrix

> Status: complete

## 题源线索

- Topic: 两组向量之间的两两欧氏距离矩阵。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

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

## 工程要点 / 面试追问

见 `notes.md`。

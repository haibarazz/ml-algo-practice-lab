# KNN Classifier

> Status: complete

## 题源线索

- Topic: 实现 KNN 距离、投票与 tie-break。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

KNN 不显式训练参数。预测时计算 query 到训练样本的距离，取最近的 `k` 个标签投票。

本模块约定 tie-break：票数相同时选择标签值较小者，保证结果确定。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def knn_classifier(X_train, y_train, X_query, k=3):
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

def knn_classifier(X_train, y_train, X_query, k=3):
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
def test_knn_classifier():
    X = np.array([[0, 0], [0, 1], [5, 5], [6, 5]])
    y = np.array([0, 0, 1, 1])
    q = np.array([[0.2, 0.1], [5.5, 5.0]])
    assert knn_classifier(X, y, q, k=3).tolist() == [0, 1]
    assert knn_classifier([[0], [2]], [1, 0], [[1]], k=2).tolist() == [0]

test_knn_classifier()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def knn_classifier(X_train, y_train, X_query, k=3):
    X_train = np.asarray(X_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    y_train = np.asarray(y_train)
    preds = []
    for q in X_query:
        dist = np.sqrt(np.sum((X_train - q) ** 2, axis=1))
        nearest = np.argsort(dist)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])
    return np.asarray(preds)
```

### 解析

1. 对每个 query 计算到所有训练样本的距离。
2. `argsort` 取最近 k 个。
3. `np.unique(..., return_counts=True)` 统计投票。
4. `np.unique` 返回有序标签，因此 `argmax` 在票数相同会选择较小标签。

## 工程要点 / 面试追问

见 `notes.md`。

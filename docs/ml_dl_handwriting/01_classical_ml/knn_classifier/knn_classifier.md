# KNN Classifier

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/knn_classifier/knn_classifier.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

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

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

### 核心公式

- $\hat y=\operatorname{mode}\{y_i: x_i \in N_k(x)\}$。
- 常见距离：欧氏距离 $\lVert x-x_i\rVert_2$，也可按任务换成曼哈顿距离或余弦距离。

### 易错点

- `k` 大于训练样本数没有处理。
- 投票 tie-break 不确定，复现实验时必须固定规则。
- 特征尺度不同会显著影响距离，通常要做标准化。
- KNN 训练很轻，但预测要扫训练集，线上延迟可能很高。

### 面试追问

::: details 参考回答：KNN 的训练复杂度和预测复杂度分别是多少？

KNN 几乎没有显式训练过程，朴素实现的训练复杂度接近存储训练集的 `O(nd)`。预测一个样本需要和所有训练样本算距离，复杂度是 `O(nd)`，预测 m 个样本就是 `O(mnd)`。

:::

::: details 参考回答：如何选择 K？K 太大或太小分别有什么问题？

K 太小会对噪声和离群点敏感，决策边界很抖；K 太大会过度平滑，少数类和局部结构容易被多数邻居淹没。通常通过交叉验证选择 K，并配合标准化和合适的距离度量。

:::

::: details 参考回答：如何加速最近邻检索，比如 KDTree、BallTree、ANN？

低维空间可以用 KDTree 或 BallTree 做精确或近似加速。高维 embedding 场景更常用 ANN 方法，比如 HNSW、IVF、PQ，用少量召回损失换取查询速度和内存效率。

:::

::: details 参考回答：KNN 为什么容易受维度灾难影响？

维度升高后，样本间距离会趋于相近，最近邻和普通邻居的区分度下降。与此同时，覆盖高维空间需要指数级更多样本，所以 KNN 的局部投票假设会变弱。

:::
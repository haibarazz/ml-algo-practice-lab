# PCA First Component

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/00_math_primitives/pca_first_component/pca_first_component.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

PCA 先对数据中心化，再求协方差矩阵的最大特征值对应特征向量。第一主成分是让投影方差最大的方向。

实现步骤：

1. 中心化 `X`。
2. 计算协方差矩阵。
3. 特征分解。
4. 取最大特征值对应向量并单位化。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def pca_first_component(X):
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

def pca_first_component(X):
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
def test_pca_first_component():
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    comp = pca_first_component(X)
    assert np.allclose(comp, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    assert np.allclose(np.linalg.norm(comp), 1.0)

test_pca_first_component()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def pca_first_component(X):
    X = np.asarray(X, dtype=np.float64)
    centered = X - X.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / X.shape[0]
    values, vectors = np.linalg.eigh(cov)
    component = vectors[:, np.argmax(values)]
    component = component / np.linalg.norm(component)
    first_nonzero = np.flatnonzero(np.abs(component) > 1e-12)
    if len(first_nonzero) and component[first_nonzero[0]] < 0:
        component = -component
    return component
```

### 解析

1. PCA 必须先中心化。
2. 协方差矩阵是对称矩阵，用 `eigh` 更合适。
3. 特征向量正负号不唯一，测试时要固定一个符号约定。


:::

## 工程要点 / 面试追问

### 核心公式

- 中心化后协方差矩阵 $C=\frac{1}{n}X_c^\top X_c$。
- 第一主成分 $v_1=\arg\max_{\lVert v\rVert=1} v^\top C v$，对应最大特征值的特征向量。

### 易错点

- 忘记中心化，第一主成分会被均值偏移污染。
- 直接用 `eig` 可能出现很小的复数数值噪声，对称矩阵优先用 `eigh` 或 SVD。
- 特征向量正负号不唯一，测试时应比较方向等价而不是固定符号。
- 不同特征量纲差异大时，是否标准化会显著影响 PCA。

### 面试追问

- 为什么第一主成分对应协方差矩阵最大特征值？
- PCA 和 SVD 的关系是什么？
- PCA 是有监督还是无监督方法？它会不会利用标签？
- PCA 降维后保留多少维通常怎么选？
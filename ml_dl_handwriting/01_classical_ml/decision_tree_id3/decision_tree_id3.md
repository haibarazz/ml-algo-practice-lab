# Decision Tree ID3

> Status: complete

## 手写实现约束

允许使用 list / dict / math；不允许调用 sklearn.tree。

## 原理最小说明

ID3 使用信息增益选特征：

$$Gain(D,A)=H(D)-\sum_v \frac{|D_v|}{|D|}H(D_v)$$

本模块只实现“选择最佳特征”这一步。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math
from collections import Counter, defaultdict

def decision_tree_id3(X, y):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import math
from collections import Counter, defaultdict

def decision_tree_id3(X, y):
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
def test_decision_tree_id3():
    X = [["sunny", "hot"], ["sunny", "cool"], ["rain", "hot"], ["rain", "cool"]]
    y = [0, 0, 1, 1]
    feature, gain = decision_tree_id3(X, y)
    assert feature == 0
    assert abs(gain - 1.0) < 1e-12

test_decision_tree_id3()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import math
from collections import Counter, defaultdict

def _entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def decision_tree_id3(X, y):
    base = _entropy(y)
    best_feature, best_gain = None, -1.0
    num_features = len(X[0])
    for j in range(num_features):
        groups = defaultdict(list)
        for row, label in zip(X, y):
            groups[row[j]].append(label)
        cond = sum(len(labels) / len(y) * _entropy(labels) for labels in groups.values())
        gain = base - cond
        if gain > best_gain:
            best_feature, best_gain = j, gain
    return best_feature, best_gain
```

### 解析

1. 先计算整体标签熵。
2. 对每个候选特征按取值分组。
3. 条件熵是各分组熵的加权平均。
4. 信息增益最大者为最佳特征。

## 工程要点 / 面试追问

### 核心公式

- $H(D)=-\sum_c p_c\log p_c$。
- $Gain(D,A)=H(D)-\sum_v\frac{|D_v|}{|D|}H(D_v)$。

### 易错点

- 条件熵忘记按子集样本占比加权。
- log 底数不影响特征排序，但会影响信息增益数值。
- 连续特征需要额外找切分点，本模块只处理离散特征。
- ID3 偏好多取值特征，工程中常用增益率或正则化限制。

### 面试追问

- ID3、C4.5、CART 的划分准则有什么区别？
- 信息增益为什么偏好多取值特征？
- 决策树如何防止过拟合？预剪枝和后剪枝有什么区别？
- Gini impurity 和 entropy 的直觉差异是什么？

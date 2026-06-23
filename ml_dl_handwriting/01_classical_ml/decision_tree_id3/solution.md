# Decision Tree ID3 Solution

## Reference Implementation

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

## Explanation

1. 先计算整体标签熵。
2. 对每个候选特征按取值分组。
3. 条件熵是各分组熵的加权平均。
4. 信息增益最大者为最佳特征。

# Binary AUC Solution

## Reference Implementation

```python
import numpy as np

def auc_binary(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires at least one positive and one negative sample")
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return total / (len(pos) * len(neg))
```

## Explanation

1. AUC 不依赖阈值，只看排序关系。
2. 平分 ties 是常见约定。
3. 如果全是正样本或全是负样本，AUC 无定义。

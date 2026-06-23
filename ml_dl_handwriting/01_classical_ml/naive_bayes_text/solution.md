# Naive Bayes Text Solution

## Reference Implementation

```python
import math
from collections import Counter, defaultdict

def naive_bayes_text(train_docs, train_labels, query, alpha=1.0):
    classes = sorted(set(train_labels))
    vocab = sorted({tok for doc in train_docs for tok in doc})
    class_counts = Counter(train_labels)
    token_counts = {c: Counter() for c in classes}
    total_tokens = defaultdict(int)
    for doc, label in zip(train_docs, train_labels):
        token_counts[label].update(doc)
        total_tokens[label] += len(doc)
    best_label, best_score = None, -float("inf")
    for c in classes:
        score = math.log(class_counts[c] / len(train_labels))
        denom = total_tokens[c] + alpha * len(vocab)
        for tok in query:
            count = token_counts[c][tok]
            score += math.log((count + alpha) / denom)
        if score > best_score:
            best_label, best_score = c, score
    return best_label
```

## Explanation

1. 统计类别先验 `P(c)`。
2. 统计每类下词频 `P(w|c)`。
3. 用 log 累加避免概率连乘下溢。
4. 未登录词在本实现中只通过平滑参与，不扩展 vocab。

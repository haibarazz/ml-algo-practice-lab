"""Lightweight tests for decision_tree_id3.

Run with: python tests.py
"""

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


def test_decision_tree_id3():
    X = [["sunny", "hot"], ["sunny", "cool"], ["rain", "hot"], ["rain", "cool"]]
    y = [0, 0, 1, 1]
    feature, gain = decision_tree_id3(X, y)
    assert feature == 0
    assert abs(gain - 1.0) < 1e-12

test_decision_tree_id3()
print("All tests passed.")

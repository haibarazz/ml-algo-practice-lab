"""Lightweight tests for naive_bayes_text.

Run with: python tests.py
"""

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


def test_naive_bayes_text():
    docs = [["good", "movie"], ["great", "good"], ["bad", "movie"], ["bad", "boring"]]
    labels = [1, 1, 0, 0]
    assert naive_bayes_text(docs, labels, ["good"]) == 1
    assert naive_bayes_text(docs, labels, ["bad"]) == 0
    assert naive_bayes_text(docs, labels, ["unknown"]) in [0, 1]

test_naive_bayes_text()
print("All tests passed.")

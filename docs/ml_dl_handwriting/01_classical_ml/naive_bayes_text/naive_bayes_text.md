# Naive Bayes Text

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/01_classical_ml/naive_bayes_text/naive_bayes_text.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 list / dict / math；不允许调用 sklearn.naive_bayes。

## 原理最小说明

多项式朴素贝叶斯假设词在类别条件下独立：

$$\log P(c|d) \propto \log P(c)+\sum_{w\in d}\log P(w|c)$$

使用 Laplace smoothing 避免未见词概率为 0。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math
from collections import Counter, defaultdict

def naive_bayes_text(train_docs, train_labels, query, alpha=1.0):
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

def naive_bayes_text(train_docs, train_labels, query, alpha=1.0):
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
def test_naive_bayes_text():
    docs = [["good", "movie"], ["great", "good"], ["bad", "movie"], ["bad", "boring"]]
    labels = [1, 1, 0, 0]
    assert naive_bayes_text(docs, labels, ["good"]) == 1
    assert naive_bayes_text(docs, labels, ["bad"]) == 0
    assert naive_bayes_text(docs, labels, ["unknown"]) in [0, 1]

test_naive_bayes_text()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

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

### 解析

1. 统计类别先验 `P(c)`。
2. 统计每类下词频 `P(w|c)`。
3. 用 log 累加避免概率连乘下溢。
4. 未登录词在本实现中只通过平滑参与，不扩展 vocab。


:::

## 工程要点 / 面试追问

### 核心公式

- $\hat y=\arg\max_c \log P(c)+\sum_j x_j\log P(w_j|c)$。
- Laplace 平滑：$P(w|c)=\frac{count(w,c)+\alpha}{\sum_v count(v,c)+\alpha |V|}$。

### 易错点

- 概率直接相乘会下溢，应在 log 空间相加。
- 未见词概率为 0，需要平滑。
- 类别先验忘记加入，类别不均衡时影响明显。
- 多项式 NB、伯努利 NB、Gaussian NB 的特征假设不同。

### 面试追问

::: details 参考回答：朴素贝叶斯“朴素”在哪里？独立性假设为什么仍常有效？

“朴素”指在给定类别后，特征之间被假设条件独立。这个假设在文本中显然不完全成立，但词频统计和类别之间的强相关信号很强，所以作为简单高偏差模型仍常有效。

:::

::: details 参考回答：多项式 NB 和伯努利 NB 的区别是什么？

多项式 NB 建模词频或计数，适合 bag-of-words 文本分类。伯努利 NB 建模词是否出现，更适合二值特征，重复出现的次数不会像多项式模型那样增加证据。

:::

::: details 参考回答：为什么文本分类里常在 log 空间计算？

文本特征维度高，很多词概率很小，直接连乘会快速下溢到 0。转到 log 空间后，概率乘法变成 log 概率加法，数值更稳定，实现也更简单。

:::

::: details 参考回答：平滑系数 alpha 太大或太小会产生什么影响？

alpha 太小会让未见词或低频词概率过低，模型对稀疏统计过敏。alpha 太大则会把分布过度拉平，削弱真实词频差异，模型变得欠拟合。

:::
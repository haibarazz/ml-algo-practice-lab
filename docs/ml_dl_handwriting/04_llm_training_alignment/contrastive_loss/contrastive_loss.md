# Contrastive Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/04_llm_training_alignment/contrastive_loss/contrastive_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

InfoNCE 把正样本放在候选集合第 0 位，对候选相似度做 cross entropy：

$$L=-\log \frac{e^{sim(q,k^+)/\tau}}{e^{sim(q,k^+)/\tau}+\sum_j e^{sim(q,k^-_j)/\tau}}$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def contrastive_loss(query, positive, negatives, temperature=1.0):
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

def contrastive_loss(query, positive, negatives, temperature=1.0):
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
def test_contrastive_loss():
    q = np.array([[1.0, 0.0]])
    p = np.array([[1.0, 0.0]])
    neg = np.array([[[0.0, 1.0], [-1.0, 0.0]]])
    good = contrastive_loss(q, p, neg, temperature=0.5)
    bad = contrastive_loss(q, np.array([[0.0, 1.0]]), neg, temperature=0.5)
    assert good < bad

test_contrastive_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def _normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def contrastive_loss(query, positive, negatives, temperature=1.0):
    q = _normalize(np.asarray(query, dtype=np.float64))
    p = _normalize(np.asarray(positive, dtype=np.float64))
    n = _normalize(np.asarray(negatives, dtype=np.float64))
    pos = np.sum(q * p, axis=-1, keepdims=True)
    neg = np.einsum('bd,bkd->bk', q, n)
    logits = np.concatenate([pos, neg], axis=1) / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    return np.mean(-log_probs[:, 0])
```

### 解析

1. 先归一化，用 cosine similarity。
2. 正样本 logit 放第 0 列。
3. 对候选集合做 cross entropy。
4. temperature 越小，分布越尖锐。


:::

## 工程要点 / 面试追问

### 核心公式

- InfoNCE：$L_i=-\log\frac{\exp(sim(q_i,k_i^+)/\tau)}{\sum_j\exp(sim(q_i,k_j)/\tau)}$。
- 若使用 cosine similarity，通常先对 embedding 做 L2 normalize。

### 易错点

- 正负样本维度对不齐，label 对应关系错位。
- 忘记 temperature，或 temperature 太小导致 softmax 过尖。
- 没归一化 embedding，点积会受向量范数影响。
- in-batch negatives 默认其他样本都是负例，但数据中可能存在 false negative。

### 面试追问

- InfoNCE 和 cross entropy 的关系是什么？
- in-batch negatives 是什么？有什么优缺点？
- temperature 如何影响训练难度和梯度？
- 为什么对比学习里常用 embedding normalize？
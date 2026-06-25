# Top-k / Top-p 采样过滤：控制生成候选集合

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/06_inference/top_k_top_p_filter/top_k_top_p_filter.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


拆解 MiniMind `generate` 中 temperature、top-k、top-p 和 multinomial 的采样流程。

## 学习目标

- 理解 logits 经过 temperature 后怎样改变分布尖锐程度。
- 掌握 top-k 和 top-p 分别如何裁剪候选 token。
- 理解采样与 greedy decoding 的差异。

## MiniMind 源码定位

- `model/model_minimind.py:257-288`

## 源码机制详解

MiniMind 每步取最后一个位置的 logits，并先除以 temperature。temperature 小于 1 会放大差异，让分布更尖；大于 1 会压平差异，让采样更随机。
top-k 过滤保留 logit 最大的 k 个 token，其余置为 `-inf`。top-p 则先按概率降序累计，只保留累计概率不超过 p 的最小高概率集合，并强制保留最高概率 token。
过滤后如果 `do_sample=True`，模型从 softmax 后的分布中 multinomial 采样；否则直接 argmax。这解释了为什么同一 prompt 在采样模式下可能有不同回答。

## 关键公式与数据流

- temperature：$p_i=softmax(z_i/T)$。
- top-k：只保留 logit 排名前 $k$ 的 token。
- top-p：保留最小集合 $S$，使 $\sum_{i\in S}p_i \ge p$。

## 为什么练这个

- 手写过滤逻辑可以理解生成参数对输出多样性的影响。
- 这也是排查模型胡言乱语、重复和过保守回答的基础。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """带提示实现。"""
    # TODO 1: 复制 logits 避免原地污染
    # TODO 2: top_k 只保留最大 k 个
    # TODO 3: top_p 按降序 softmax 累计过滤
    # TODO 4: 至少保留最高概率 token
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """无提示实现。"""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现无提示函数后直接运行测试区代码。

```python
import numpy as np


def test_top_k_top_p_filter():
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    out = top_k_top_p_filter(logits, top_k=2, top_p=1.0)
    assert np.isneginf(out[2]) and np.isneginf(out[3])
    out = top_k_top_p_filter(logits, top_k=0, top_p=0.7)
    assert not np.isneginf(out[0])
    assert np.isneginf(out[-1])


test_top_k_top_p_filter()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    logits = np.asarray(logits, dtype=np.float64).copy()
    if top_k > 0:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf
    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        finite = np.isfinite(sorted_logits)
        shifted = sorted_logits[finite] - np.max(sorted_logits[finite])
        probs = np.exp(shifted) / np.exp(shifted).sum()
        remove = np.zeros_like(sorted_logits, dtype=bool)
        finite_indices = np.where(finite)[0]
        cumulative = np.cumsum(probs)
        remove[finite_indices] = cumulative > top_p
        if len(remove) > 0:
            remove[0] = False
        logits[order[remove]] = -np.inf
    return logits
```

### 解析

1. MiniMind 的实现会把超过 top-p 的 mask 右移，保证第一个 token 保留。
2. top_k 和 top_p 同时开时，top_k 先执行。


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- temperature：$p_i=softmax(z_i/T)$。
- top-k：只保留 logit 排名前 $k$ 的 token。
- top-p：保留最小集合 $S$，使 $\sum_{i\in S}p_i \ge p$。

### 易错点

- top-p mask 要右移，保证第一个超过阈值的 token 被保留。
- 所有 token 都被过滤会导致 softmax NaN。
- temperature 不能为 0；greedy 应用 argmax 表达。

### 面试追问

::: details 参考回答：top-k 和 top-p 的核心区别是什么？

top-k 固定保留 token 数，不管分布是否尖锐；top-p 固定保留累计概率质量，候选数量会随分布形状变化。top-p 在不确定时允许更多候选，在确定时更保守。

:::

::: details 参考回答：temperature 为什么能控制创造性？

temperature 改变 logits 差异。低温让高概率 token 更占优势，输出稳定；高温让低概率 token 更有机会被采样，输出更多样但也更容易不稳定。

:::
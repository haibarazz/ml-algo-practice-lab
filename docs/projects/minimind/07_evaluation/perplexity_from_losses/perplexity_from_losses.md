# Perplexity from Losses

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/07_evaluation/perplexity_from_losses/perplexity_from_losses.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `eval_llm.py:12-93`
- `model/model_minimind.py:245-253`

## 手写实现约束

只用 math/list。

## 原理最小说明

语言模型常用 loss 的指数作为 perplexity。多个 batch 时应先按 token 数加权平均 loss，再取 exp。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math


def perplexity_from_losses(losses, token_counts):
    """TODO guided implementation."""
    # TODO 1: 计算 loss*token_count 的总和
    # TODO 2: 除以 token 总数
    # TODO 3: 返回 exp(mean_loss)
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import math


def perplexity_from_losses(losses, token_counts):
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
import math


def test_perplexity_from_losses():
    ppl = perplexity_from_losses([math.log(2), math.log(4)], [10, 10])
    assert abs(ppl - math.sqrt(8)) < 1e-12
    assert perplexity_from_losses([0.0], [5]) == 1.0


test_perplexity_from_losses()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import math

def perplexity_from_losses(losses, token_counts):
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise ValueError("token_counts must sum to a positive value")
    mean_loss = sum(l * n for l, n in zip(losses, token_counts)) / total_tokens
    return math.exp(mean_loss)
```

### 解析

1. 不能简单平均 batch loss，除非每个 batch token 数一样。
2. perplexity 越低通常表示语言建模越好。


:::

## 工程要点 / 面试追问

### Source Mapping

- `eval_llm.py:12-93`
- `model/model_minimind.py:245-253`

### 常见坑

- 不能简单平均 batch loss，除非每个 batch token 数一样。
- perplexity 越低通常表示语言建模越好。

### 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？
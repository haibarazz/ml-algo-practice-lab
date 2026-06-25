# 困惑度：从平均交叉熵到 PPL

把语言模型 loss 转成更直观的 perplexity，并理解它的适用边界。

## 学习目标

- 理解 PPL 与 token 平均 cross entropy 的关系。
- 知道 PPL 只能在同 tokenizer、同数据分布下比较。
- 理解 ignore token 会影响平均分母。

## MiniMind 源码定位

- `model/model_minimind.py:245-253`
- `eval_llm.py`

## 源码机制详解

MiniMind 的训练 loss 是按有效 token 计算的 cross entropy。若这个 CE 是自然对数下的平均负 log likelihood，那么 perplexity 就是 `exp(CE)`。
PPL 可以理解为模型在每个位置平均还困惑于多少个等概率候选。CE 越低，真实 token 的平均概率越高，PPL 越低。
但 PPL 很依赖 tokenizer、数据集和 mask 规则。不同词表会改变 token 切分，同一段文本 token 数不同；SFT 中只算 assistant token，和预训练全 token PPL 也不能直接比较。

## 关键公式与数据流

- $CE=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t|x_{<t})$。
- $PPL=\exp(CE)$。
- 若使用以 2 为底的 log，则 $PPL=2^{CE_2}$。

## 为什么练这个

- 手写 PPL 计算能把训练日志里的 loss 转成可解释指标。
- 这个模块也提醒你不要跨 tokenizer 或跨 mask 规则比较 PPL。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import math


def perplexity_from_losses(losses, token_counts):
    """带提示实现。"""
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
import math


def test_perplexity_from_losses():
    ppl = perplexity_from_losses([math.log(2), math.log(4)], [10, 10])
    assert abs(ppl - math.sqrt(8)) < 1e-12
    assert perplexity_from_losses([0.0], [5]) == 1.0


test_perplexity_from_losses()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

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

## 工程要点 / 面试追问

### 关键公式与数据流

- $CE=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t|x_{<t})$。
- $PPL=\exp(CE)$。
- 若使用以 2 为底的 log，则 $PPL=2^{CE_2}$。

### 易错点

- 用 batch loss 的平均再平均时，要确认每个 batch 有效 token 数是否相同。
- 把 pad 或 prompt token 计入分母，会让指标含义变化。
- PPL 低不代表指令跟随、事实性或安全性一定好。

### 面试追问

::: details 参考回答：PPL 为什么是 `exp(loss)`？

因为 cross entropy 是平均负 log 概率。取指数后回到概率空间，可以解释为模型平均每步的有效候选数。

:::

::: details 参考回答：什么时候 PPL 不适合比较两个模型？

当 tokenizer、评测文本、mask 规则或上下文长度不同的时候，PPL 不再是同一分布下的同一指标。特别是聊天 SFT 和预训练 PPL 不能简单横向比较。

:::

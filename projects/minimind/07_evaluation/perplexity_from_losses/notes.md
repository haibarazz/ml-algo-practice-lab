# 困惑度：从平均交叉熵到 PPL笔记

## 关键公式与数据流

- $CE=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t|x_{<t})$。
- $PPL=\exp(CE)$。
- 若使用以 2 为底的 log，则 $PPL=2^{CE_2}$。

## 易错点

- 用 batch loss 的平均再平均时，要确认每个 batch 有效 token 数是否相同。
- 把 pad 或 prompt token 计入分母，会让指标含义变化。
- PPL 低不代表指令跟随、事实性或安全性一定好。

## 面试追问

::: details 参考回答：PPL 为什么是 `exp(loss)`？

因为 cross entropy 是平均负 log 概率。取指数后回到概率空间，可以解释为模型平均每步的有效候选数。

:::

::: details 参考回答：什么时候 PPL 不适合比较两个模型？

当 tokenizer、评测文本、mask 规则或上下文长度不同的时候，PPL 不再是同一分布下的同一指标。特别是聊天 SFT 和预训练 PPL 不能简单横向比较。

:::

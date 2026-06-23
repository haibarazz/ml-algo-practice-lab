# Causal LM Loss Notes

## 核心公式

- $L=-\frac{1}{N}\sum_t \log p_\theta(x_{t+1}|x_{\le t})$。
- $PPL=\exp(CE)$，其中 CE 是按 token 平均的交叉熵。

## 易错点

- 忘记 shift，变成预测当前 token。
- `ignore_index` 没处理，pad token 被计入 loss。
- logits/labels flatten 后顺序对不齐。
- loss 平均分母应只统计未被 mask 的有效 token。

## 面试追问

::: details 参考回答：causal LM 为什么要把 logits 和 labels shift 一位？

causal LM 在位置 t 的隐藏状态只能预测下一个 token `x_{t+1}`，所以 logits 和 labels 要错开一位。若不 shift，就会变成用当前位置预测自己，训练目标不符合自回归生成。

:::

::: details 参考回答：perplexity 和 cross entropy loss 的关系是什么？

perplexity 等于按 token 平均 cross entropy 的指数，即 `PPL = exp(CE)`。CE 越低，模型给真实下一个 token 的平均概率越高，PPL 也越低。

:::

::: details 参考回答：训练时 pad token 为什么要 ignore？

pad token 是为了 batch 对齐添加的，不是真实文本内容。把它计入 loss 会让模型学习预测 padding，并且不同样本长度会不公平地影响训练目标。

:::

::: details 参考回答：SFT 为什么通常只对 answer token 计 loss？

SFT 的目标通常是让模型在给定 prompt 后生成高质量 answer，而不是复述 prompt。只对 answer token 计 loss，可以把监督信号集中在助手回复上，避免用户输入本身污染训练。

:::

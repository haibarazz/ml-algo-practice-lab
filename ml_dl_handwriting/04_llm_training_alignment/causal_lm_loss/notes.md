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

- causal LM 为什么要把 logits 和 labels shift 一位？
- perplexity 和 cross entropy loss 的关系是什么？
- 训练时 pad token 为什么要 ignore？
- SFT 为什么通常只对 answer token 计 loss？

# Causal LM Loss Notes

## 易错点
- 忘记 shift，变成预测当前 token。
- ignore_index 没处理。
- logits/labels flatten 对不齐。

## 面试追问
- perplexity 和 CE loss 的关系？
- SFT 为什么要 mask prompt？

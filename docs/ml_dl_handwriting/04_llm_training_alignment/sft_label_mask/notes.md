# SFT Label Mask Notes

## 易错点
- 把 prompt 也计入 loss。
- pad 没 mask。
- prompt_lengths 和 batch 对不齐。

## 面试追问
- 为什么只训练 answer token？
- 多轮对话里 mask 如何设计？

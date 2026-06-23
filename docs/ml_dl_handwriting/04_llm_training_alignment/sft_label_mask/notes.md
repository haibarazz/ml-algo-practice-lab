# SFT Label Mask Notes

## 核心公式

- $L=-\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}\log p_\theta(y_t|y_{<t},prompt)$。
- prompt/pad 位置 label 通常设为 `ignore_index`，只让 answer token 贡献 loss。

## 易错点

- 把 prompt 也计入 loss，模型会被训练去复述用户输入。
- pad 没 mask，batch padding 会污染 loss。
- `prompt_lengths` 和 batch 样本对不齐。
- 多轮对话中 assistant/user/system 边界没有明确标注。

## 面试追问

- SFT 为什么通常只训练 answer token？
- 多轮对话中 label mask 应如何设计？
- 如果把 prompt token 也计入 loss，会带来什么问题？
- 不同 chat template 会怎样影响 mask 边界？

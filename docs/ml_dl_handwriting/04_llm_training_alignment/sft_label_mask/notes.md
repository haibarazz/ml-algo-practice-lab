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

::: details 参考回答：SFT 为什么通常只训练 answer token？

SFT 训练的是在 prompt 条件下生成 answer 的能力，prompt 是条件，不是要模型学习输出的目标。只训练 answer token 可以避免模型被奖励去复述用户问题或 system 指令。

:::

::: details 参考回答：多轮对话中 label mask 应如何设计？

多轮对话中通常只让 assistant 回复位置参与 loss，system/user token 和 padding 都设为 ignore。若要训练特定轮次，也要严格根据 chat template 标记每段角色边界。

:::

::: details 参考回答：如果把 prompt token 也计入 loss，会带来什么问题？

prompt token 也计入 loss 会让模型优化“预测用户输入”的概率，而这不是部署时的生成任务。它还会让长 prompt 样本贡献过多 loss，稀释 answer 部分的监督信号。

:::

::: details 参考回答：不同 chat template 会怎样影响 mask 边界？

chat template 决定 system/user/assistant 的特殊 token、分隔符和起止位置。模板不同会改变 answer 从哪里开始、哪些 token 属于角色标记，因此 label mask 边界必须和模板一致。

:::

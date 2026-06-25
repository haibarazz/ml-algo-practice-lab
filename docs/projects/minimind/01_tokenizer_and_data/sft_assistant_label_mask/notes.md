# SFT 标签掩码：只训练 assistant 回复笔记

## 关键公式与数据流

- SFT 目标：$L=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log p_\theta(x_t|x_{<t})$。
- $\mathcal A$ 是 assistant token 位置集合；非 assistant、pad、system、user 位置 label 设为 $-100$。

## 易错点

- assistant 起始 token 少了换行，mask 会完全扫不到。
- 把 user/system 也计入 loss，会稀释回答部分监督，还会鼓励模型复述输入。
- 不同 chat template 的角色边界不同，不能复用硬编码 token 序列。

## 面试追问

::: details 参考回答：SFT 为什么通常只对 assistant token 计 loss？

部署时模型的任务是在 prompt 条件下生成 assistant 回复，prompt 是条件而不是目标。只训练 assistant token 能把梯度集中到回答行为上，避免模型学习复述用户输入。

:::

::: details 参考回答：多轮对话里 label mask 的边界最容易错在哪里？

最容易错在 role special token 和换行。真实 tokenizer 看到的是模板渲染后的 token 序列，必须用 tokenizer 编码出来的 assistant 起止片段匹配，而不是靠字符串长度估算。

:::

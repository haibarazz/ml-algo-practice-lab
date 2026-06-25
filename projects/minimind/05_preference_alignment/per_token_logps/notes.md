# 逐 token logprob：rollout 后如何评估生成概率笔记

## 关键公式与数据流

- $\log p_\theta(y_t|x,y_{<t})=\log softmax(logits_{t-1})[y_t]$。
- $ratio_t=\exp(\log p_\theta(y_t)-\log p_{old}(y_t))$。

## 易错点

- `n_keep` 少加 1 会拿不到预测第一个生成 token 的 logits。
- gather 维度必须是 vocab 维。
- 生成序列里的 pad token 需要后续 completion_mask 屏蔽。

## 面试追问

::: details 参考回答：为什么 rollout 后不直接使用 generate 时的概率？

有些推理后端不返回完整概率，或者需要用当前训练图重新计算可对齐的 logprob。重新 forward 可以确保 logprob 与当前 policy、mask 和 dtype 处理一致。

:::

::: details 参考回答：per-token logprob 和序列 logprob 有什么关系？

序列 logprob 通常是有效 token logprob 的和。PPO/GRPO 常保留逐 token 形式，因为 ratio、clip 和 mask 都可能按 token 粒度计算。

:::

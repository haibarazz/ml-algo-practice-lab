# Top-k / Top-p 采样过滤：控制生成候选集合笔记

## 关键公式与数据流

- temperature：$p_i=softmax(z_i/T)$。
- top-k：只保留 logit 排名前 $k$ 的 token。
- top-p：保留最小集合 $S$，使 $\sum_{i\in S}p_i \ge p$。

## 易错点

- top-p mask 要右移，保证第一个超过阈值的 token 被保留。
- 所有 token 都被过滤会导致 softmax NaN。
- temperature 不能为 0；greedy 应用 argmax 表达。

## 面试追问

::: details 参考回答：top-k 和 top-p 的核心区别是什么？

top-k 固定保留 token 数，不管分布是否尖锐；top-p 固定保留累计概率质量，候选数量会随分布形状变化。top-p 在不确定时允许更多候选，在确定时更保守。

:::

::: details 参考回答：temperature 为什么能控制创造性？

temperature 改变 logits 差异。低温让高概率 token 更占优势，输出稳定；高温让低概率 token 更有机会被采样，输出更多样但也更容易不稳定。

:::

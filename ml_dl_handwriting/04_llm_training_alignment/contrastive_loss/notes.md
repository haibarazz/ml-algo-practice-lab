# Contrastive Loss Notes

## 核心公式

- InfoNCE：$L_i=-\log\frac{\exp(sim(q_i,k_i^+)/\tau)}{\sum_j\exp(sim(q_i,k_j)/\tau)}$。
- 若使用 cosine similarity，通常先对 embedding 做 L2 normalize。

## 易错点

- 正负样本维度对不齐，label 对应关系错位。
- 忘记 temperature，或 temperature 太小导致 softmax 过尖。
- 没归一化 embedding，点积会受向量范数影响。
- in-batch negatives 默认其他样本都是负例，但数据中可能存在 false negative。

## 面试追问

::: details 参考回答：InfoNCE 和 cross entropy 的关系是什么？

InfoNCE 可以看成一个 cross entropy：给定 query，要在一个正样本和多个负样本中分类出正确 key。分子是正样本 logit，分母是所有候选 logit 的 softmax 归一化。

:::

::: details 参考回答：in-batch negatives 是什么？有什么优缺点？

in-batch negatives 指一个 batch 中其他样本的 key 被当作当前 query 的负样本。优点是不用额外采样就能得到很多负例，缺点是 batch 内可能存在语义相近的 false negative。

:::

::: details 参考回答：temperature 如何影响训练难度和梯度？

temperature 越小，softmax 越尖锐，模型更关注难负样本，但梯度也可能更不稳定。temperature 越大，分布更平滑，训练信号更温和，但区分压力可能不足。

:::

::: details 参考回答：为什么对比学习里常用 embedding normalize？

normalize 后点积等价于 cosine similarity，训练更关注方向而不是向量范数。这样可以避免模型通过单纯放大 embedding 范数来提高相似度，使表示空间更稳定。

:::

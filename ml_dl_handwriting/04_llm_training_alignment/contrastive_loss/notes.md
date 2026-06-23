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

- InfoNCE 和 cross entropy 的关系是什么？
- in-batch negatives 是什么？有什么优缺点？
- temperature 如何影响训练难度和梯度？
- 为什么对比学习里常用 embedding normalize？

# Preference Pair Format Notes

## 核心公式

- 偏好样本通常包含同一个 prompt 下的一对回答：`(prompt, chosen, rejected)`。
- pairwise 学习关注 $\log p(chosen|prompt)-\log p(rejected|prompt)$ 的相对差异。

## 易错点

- `chosen`/`rejected` 顺序写反，训练信号会完全相反。
- prompt 没保留，后续无法拼接输入或计算条件概率。
- chosen/rejected 使用不同 chat template，导致比较不公平。
- 长度差异过大时，sum logprob 和 mean logprob 的选择会影响偏好强度。

## 面试追问

::: details 参考回答：DPO 数据和 SFT 数据有什么区别？

SFT 数据通常是单个 prompt 对应一个示范答案，用最大似然学习“模仿这个答案”。DPO 数据是同一 prompt 下的 chosen/rejected 成对偏好，用相对 logprob 学习更偏向 chosen。

:::

::: details 参考回答：pairwise preference 数据如何构造和清洗？

构造时要保证 chosen 和 rejected 回答同一个 prompt，并且偏好标签可靠。清洗时要去掉顺序反转、模板不一致、重复样本、过短无效回答和明显长度偏置过强的样本。

:::

::: details 参考回答：为什么 chosen/rejected 必须共享同一个 prompt？

共享 prompt 才能把比较焦点放在回答质量上，而不是不同问题本身的难度差异。若 prompt 不同，chosen/rejected 的 logprob 差就混入了条件差异，偏好信号不再干净。

:::

::: details 参考回答：偏好数据中长度偏置会怎样影响训练？

如果 chosen 系统性更长或更短，用 sum logprob 会偏向长度相关的概率差异，而不一定是质量差异。常见处理包括只比较 answer token、检查长度分布、使用 mean logprob 或做长度分桶分析。

:::

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

- DPO 数据和 SFT 数据有什么区别？
- pairwise preference 数据如何构造和清洗？
- 为什么 chosen/rejected 必须共享同一个 prompt？
- 偏好数据中长度偏置会怎样影响训练？

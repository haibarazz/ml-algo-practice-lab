# DPO 偏好样本：chosen/rejected 的 x、y、mask笔记

## 关键公式与数据流

- $x = ids_{0:T-1}$，$y = ids_{1:T}$，$mask = response\_mask_{1:T}$。
- $\log p_\theta(y|x)=\sum_t mask_t \log p_\theta(y_t|x_{\le t})$。
- chosen/rejected 必须来自同一个 prompt，否则 logprob 差混入了问题难度差异。

## 易错点

- chosen/rejected 顺序反了，训练目标会鼓励坏回答。
- mask 没有跟着 shift，logprob 会统计错 token。
- chosen/rejected 使用不同模板，会让偏好比较不公平。

## 面试追问

::: details 参考回答：DPO 数据和 SFT 数据最核心的区别是什么？

SFT 是单个 prompt 对一个示范答案，目标是模仿；DPO 是同一 prompt 下 chosen/rejected 成对比较，目标是提高 chosen 相对 rejected 的概率优势。

:::

::: details 参考回答：为什么 DPO 里 chosen 和 rejected 要共享同一个 prompt？

只有共享 prompt，模型比较的才是回答质量差异。prompt 不同会把问题难度、长度和主题差异混进 logprob margin，偏好信号会变脏。

:::

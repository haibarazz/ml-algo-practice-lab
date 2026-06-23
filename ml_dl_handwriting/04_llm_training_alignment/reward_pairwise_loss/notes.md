# Reward Pairwise Loss Notes

## 核心公式

- Bradley-Terry 形式：$P(y_w \succ y_l)=\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))$。
- pairwise loss：$L=-\log\sigma(r_w-r_l)$。

## 易错点

- 用 MSE 拟合绝对分数，而不是学习相对偏好。
- `chosen`/`rejected` 顺序反了。
- reward 输出尺度无约束，后续 RL 或排序时需要注意校准和 clipping。
- 只看 pair accuracy 不够，还要关注 reward hacking 和分布外泛化。

## 面试追问

::: details 参考回答：Reward model 为什么可以只学相对偏好？

偏好标注通常只可靠地表达“哪个回答更好”，不一定给出可比较的绝对分数。pairwise loss 学习的是 reward 差值，只要 chosen 分数高于 rejected 就符合监督信号。

:::

::: details 参考回答：pairwise、pointwise、listwise ranking loss 有什么区别？

pointwise loss 给单个样本拟合绝对标签或分数，pairwise loss 比较一对样本的相对顺序。listwise loss 一次考虑一个列表的整体排序，信息更丰富但数据和实现更复杂。

:::

::: details 参考回答：reward 分数的绝对值有意义吗？

reward 的绝对值通常没有稳定语义，因为加一个常数不改变 pairwise 偏好概率。更重要的是同一 prompt 或同一分布下的相对排序，以及分数尺度在后续 RL 中是否稳定。

:::

::: details 参考回答：reward model 如何被用于 PPO/RLHF？

在 PPO/RLHF 中，reward model 给模型生成的回答打分，作为强化学习的奖励信号。训练策略时通常还会加入 KL 惩罚，避免模型为了高 reward 偏离参考模型太远。

:::

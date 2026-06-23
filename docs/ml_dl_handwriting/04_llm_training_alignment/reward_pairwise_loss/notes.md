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

- Reward model 为什么可以只学相对偏好？
- pairwise、pointwise、listwise ranking loss 有什么区别？
- reward 分数的绝对值有意义吗？
- reward model 如何被用于 PPO/RLHF？

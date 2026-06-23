# DPO Loss Notes

## 核心公式

- $L_{DPO}=-\log\sigma\left(\beta[(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))]\right)$。
- 其中 $y_w$ 是 chosen，$y_l$ 是 rejected，$\beta$ 控制相对 reference 的约束强度。

## 易错点

- `chosen`/`rejected` 顺序写反，loss 会鼓励坏答案。
- 忘记减 reference margin，退化成简单偏好分类。
- sigmoid/log 分开算容易数值不稳定，应使用 log-sigmoid。
- logprob 应只统计回答 token，prompt token 不应影响偏好差。

## 面试追问

- DPO 和 PPO/RLHF 的主要区别是什么？
- DPO 里 reference model 起什么作用？
- beta 变大或变小会怎样影响训练？
- DPO 为什么可以看作不显式训练 reward model 的偏好优化？

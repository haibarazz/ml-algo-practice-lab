# DPO Loss Notes

## 易错点
- chosen/rejected 顺序写反。
- 忘记减 reference margin。
- sigmoid/log 分开算导致数值不稳定。

## 面试追问
- DPO 和 PPO/RLHF 的区别？
- beta 变大有什么影响？

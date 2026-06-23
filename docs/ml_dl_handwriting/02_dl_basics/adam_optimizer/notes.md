# Adam Optimizer Notes

## 易错点
- 忘记 bias correction。
- t 从 0 开始导致除零。
- v 用 grad 而不是 grad^2。

## 面试追问
- Adam 和 SGD momentum 的区别？
- AdamW 为什么把 weight decay 解耦？

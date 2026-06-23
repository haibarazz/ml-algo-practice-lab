# BatchNorm Backward Notes

## 易错点
- 求和 axis 写错。
- 忘记乘 gamma。
- eps 只在 forward 加，backward 也必须使用同一个 var+eps。

## 面试追问
- 手推 BN backward 的关键链路？
- BN 对 batch size 敏感吗？

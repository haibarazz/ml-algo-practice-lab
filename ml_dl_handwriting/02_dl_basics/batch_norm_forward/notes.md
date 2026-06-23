# BatchNorm Forward Notes

## 易错点
- 把 LayerNorm 的 axis 用到 BatchNorm。
- 忘记 eps。
- 推理态应使用 running stats，本模块只做训练态。

## 面试追问
- BN 训练和推理有什么不同？
- 为什么 Transformer 更常用 LayerNorm？

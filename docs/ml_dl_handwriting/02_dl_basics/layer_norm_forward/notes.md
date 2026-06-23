# LayerNorm Forward Notes

## 易错点
- 和 BatchNorm 混淆 axis。
- 忘记 keepdims。
- gamma/beta shape 无法广播。

## 面试追问
- Pre-LN 和 Post-LN Transformer 区别？
- 为什么 LLM 常用 LayerNorm/RMSNorm？

# SwiGLU Notes

## 核心公式

- $SiLU(x)=x\sigma(x)$。
- $SwiGLU(x)=(SiLU(xW_g)\odot xW_u)W_d$。

## 易错点

- gate 和 up projection 的隐藏维度不一致。
- 用 sigmoid 直接代替 SiLU，变成普通 GLU 变体。
- 忘记 down projection，输出维度无法回到 `d_model`。
- SwiGLU 参数量通常比普通 FFN 大，hidden size 需要相应调整。

## 面试追问

- GLU、GeGLU、SwiGLU 的区别是什么？
- 为什么现代 LLM 常用 gated FFN？
- SwiGLU 相比 ReLU/GELU FFN 增加了哪些参数？
- SiLU gate 的输出为什么可以看作数据依赖的通道筛选？

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

::: details 参考回答：GLU、GeGLU、SwiGLU 的区别是什么？

GLU 用一个 sigmoid gate 控制另一个线性分支，GeGLU 把 gate 激活换成 GELU，SwiGLU 换成 SiLU。它们共同点是用门控机制做通道筛选，差别在 gate 的非线性形式。

:::

::: details 参考回答：为什么现代 LLM 常用 gated FFN？

gated FFN 能根据输入动态调节哪些通道被放大或抑制，比普通激活更像数据依赖的特征选择。现代 LLM 追求更强表达和稳定训练，因此常用 SwiGLU/GeGLU 替代 ReLU/GELU FFN。

:::

::: details 参考回答：SwiGLU 相比 ReLU/GELU FFN 增加了哪些参数？

SwiGLU 通常有 gate projection、up projection 和 down projection 三组线性层，而普通 FFN 只有 up 和 down 两组。为了让参数量接近普通 FFN，SwiGLU 的中间维度常会设得小一些。

:::

::: details 参考回答：SiLU gate 的输出为什么可以看作数据依赖的通道筛选？

SiLU gate 的值由输入本身决定，并且是连续平滑的缩放因子。它不是简单保留或丢弃通道，而是按样本内容动态调整每个隐藏通道的贡献。

:::

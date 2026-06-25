# SwiGLU FFN：门控前馈网络笔记

## 关键公式与数据流

- $SwiGLU(x)=W_d\left(SiLU(W_gx)\odot W_ux\right)$。
- $SiLU(z)=z\sigma(z)$。
- 参数主要来自 $W_g,W_u,W_d$ 三个矩阵。

## 易错点

- 把 gate 分支写成普通 sigmoid GLU，会和 MiniMind 的 hidden_act 不一致。
- 忘记 down projection 会让输出维度无法接回残差。
- gate/up 维度不一致时逐元素乘法会隐式广播或直接报错。

## 面试追问

::: details 参考回答：SwiGLU 为什么常见于现代 LLM？

它用数据依赖的门控控制隐藏通道，比单一激活函数更灵活。经验上在相近计算预算下，gated FFN 往往比 ReLU/GELU FFN 有更好的表达和训练效果。

:::

::: details 参考回答：FFN 在 Transformer 中起什么作用？

attention 负责 token 间信息交换，FFN 负责对每个 token 的表示做非线性变换。它是逐 token 独立计算的，但参数量通常占整个 block 的很大比例。

:::

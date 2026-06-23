# RoPE Apply Notes

## 核心公式

- 二维配对旋转：$[x_{2i},x_{2i+1}] \mapsto [x_{2i}\cos\theta-x_{2i+1}\sin\theta,\;x_{2i}\sin\theta+x_{2i+1}\cos\theta]$。
- RoPE 通常作用于 Q/K，使 attention score 依赖相对位置。

## 易错点

- 奇偶维配对错，导致旋转不成对。
- sin/cos 广播维度错，batch/head/seq 维对不上。
- 对 V 也应用 RoPE；通常 RoPE 用于 Q/K。
- position index 从 0 还是从 offset 开始，在 KV cache 推理中必须一致。

## 面试追问

- RoPE 为什么能表达相对位置信息？
- RoPE 和绝对位置编码有什么区别？
- RoPE 在 KV cache 推理时 position offset 如何处理？
- RoPE 和 ALiBi 的设计直觉有什么不同？

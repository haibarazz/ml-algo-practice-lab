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

::: details 参考回答：RoPE 为什么能表达相对位置信息？

RoPE 对 Q/K 的成对维度按位置角度旋转，两个位置的内积会自然包含相对位置差。也就是说，attention score 不只看内容相似度，还能感知 query 和 key 的位置间隔。

:::

::: details 参考回答：RoPE 和绝对位置编码有什么区别？

绝对位置编码通常是把位置向量加到 token embedding 上，位置信息混在输入表示里。RoPE 不直接加向量，而是在注意力计算前旋转 Q/K，使相对位置信息进入点积。

:::

::: details 参考回答：RoPE 在 KV cache 推理时 position offset 如何处理？

KV cache 推理时，新 token 的 position 不能从 0 重新开始，而要接在已缓存长度之后。否则新 Q 和旧 K 使用的位置坐标不一致，attention 的相对位置会错乱。

:::

::: details 参考回答：RoPE 和 ALiBi 的设计直觉有什么不同？

RoPE 用旋转让点积编码相对位置，属于乘性或几何式位置注入。ALiBi 直接按距离给 attention logits 加负偏置，鼓励近处更高权重，结构更简单且外推直觉更直接。

:::

# RoPE 旋转位置编码：MiniMind 的 rotate_half 写法笔记

## 关键公式与数据流

- 二维旋转：$(x_1,x_2)\mapsto(x_1\cos\theta-x_2\sin\theta,\ x_1\sin\theta+x_2\cos\theta)$。
- MiniMind 写法：$rope(x)=x\odot cos + rotate\_half(x)\odot sin$。
- RoPE 作用于 Q/K，使 attention score 依赖相对位置差。

## 易错点

- 偶奇配对和前后半维配对是两种实现约定，不能混用。
- cos/sin 的 seq 维、head 维广播错，会导致 batch 测试不稳定。
- KV cache 下 position 从 0 重启，会让新旧 token 的相对位置错乱。

## 面试追问

::: details 参考回答：RoPE 为什么能表达相对位置信息？

同一旋转矩阵族作用到 Q/K 后，两者点积会出现位置角度差项。也就是说 attention score 不只取决于内容向量，还取决于 query 和 key 的相对距离。

:::

::: details 参考回答：为什么 MiniMind 对 Q/K 旋转，而不是对 V 旋转？

attention 权重由 Q/K 点积决定，位置信息需要影响“看谁”。V 是被权重加权求和的内容载体，通常不需要承担位置匹配。

:::

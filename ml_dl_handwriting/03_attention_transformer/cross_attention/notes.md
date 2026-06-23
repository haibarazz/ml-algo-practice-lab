# Cross Attention Notes

## 核心公式

- cross-attention 中 $Q$ 来自当前解码端/查询序列，$K,V$ 来自上下文序列。
- $O=softmax(QK_{ctx}^\top/\sqrt{d_k})V_{ctx}$，输出长度等于 query 长度。

## 易错点

- 误以为输出长度等于 context 长度；实际由 Q 的长度决定。
- Q/K/V 投影来源写混，导致 encoder-decoder 语义错误。
- mask 要区分 query mask、key padding mask 和 causal mask。
- context 长度很长时，cross-attention 也会带来较高显存成本。

## 面试追问

- encoder-decoder Transformer 哪些位置使用 cross-attention？
- cross-attention 和 self-attention 在 Q/K/V 来源上有什么区别？
- RAG 或多模态模型中 cross-attention 的直觉是什么？
- cross-attention 的输出长度由什么决定？

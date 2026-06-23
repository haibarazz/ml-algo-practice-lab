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

::: details 参考回答：encoder-decoder Transformer 哪些位置使用 cross-attention？

encoder-decoder Transformer 中，decoder 的每一层通常先做 masked self-attention，再通过 cross-attention 读取 encoder 输出。机器翻译、摘要等 seq2seq 模型里，这一步负责把目标端生成和源端表示对齐。

:::

::: details 参考回答：cross-attention 和 self-attention 在 Q/K/V 来源上有什么区别？

self-attention 的 Q/K/V 来自同一序列，建模序列内部关系。cross-attention 的 Q 来自当前查询序列，K/V 来自外部上下文，因此本质是“用 query 去检索 context”。

:::

::: details 参考回答：RAG 或多模态模型中 cross-attention 的直觉是什么？

在 RAG 中，cross-attention 可以让生成端按需读取检索文档表示；在多模态中，文本 query 可以读取图像或音频 token。直觉上它是一个可学习的软检索和信息融合模块。

:::

::: details 参考回答：cross-attention 的输出长度由什么决定？

输出长度由 Q 的长度决定，因为每个 query 位置都会得到一个加权后的 value 表示。K/V 只决定可被读取的上下文范围和注意力分布的 key 维度。

:::

# Multi Head Attention Notes

## 核心公式

- $head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$。
- $MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W^O$。

## 易错点

- split/transpose 维度写错，head 维和 seq 维互换。
- concat 轴写错，导致输出 embedding 维不对。
- 忘记输出投影 $W^O$。
- `d_model` 不能被 `num_heads` 整除时需要报错。

## 面试追问

::: details 参考回答：多头为什么有用？它和单头大维度 attention 有什么差异？

多头把表示拆到多个子空间，每个头学习一套注意力模式，最后再拼接融合。单头大维度虽然容量大，但只有一张注意力图，多头能并行表达多种关系。

:::

::: details 参考回答：MHA 参数量怎么估算？

标准 MHA 中 Q/K/V 投影各有 `d_model * d_model` 参数，输出投影也有 `d_model * d_model`，总量约 `4 * d_model^2`，不含 bias。num_heads 改变的是拆分方式，通常不改变总投影参数量。

:::

::: details 参考回答：head_dim 为什么通常设为 `d_model / num_heads`？

`head_dim = d_model / num_heads` 可以让拼接后维度回到 `d_model`，同时保持总计算量可控。这样增加头数主要改变注意力分解粒度，而不是线性增大 hidden 宽度。

:::

::: details 参考回答：推理时 MHA 的主要瓶颈在哪里？

推理时主要瓶颈是 attention 读写 KV cache 的显存带宽，以及长上下文下的 QK 计算。batch、层数、head 数和序列长度都会放大这个成本。

:::

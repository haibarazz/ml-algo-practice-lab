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

- 多头为什么有用？它和单头大维度 attention 有什么差异？
- MHA 参数量怎么估算？
- head_dim 为什么通常设为 `d_model / num_heads`？
- 推理时 MHA 的主要瓶颈在哪里？

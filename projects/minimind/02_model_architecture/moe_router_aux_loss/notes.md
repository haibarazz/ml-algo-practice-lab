# MoE 路由辅助损失：让专家负载更均衡笔记

## 关键公式与数据流

- $p(e|x)=softmax(W_rx)$，选择 top-k expert。
- $y_i=\sum_{e\in topk(i)} w_{i,e} Expert_e(x_i)$。
- $L_{aux}=coef \cdot E \cdot \sum_e load_e \cdot score_e$。

## 易错点

- top-k 概率不归一化时，多 expert 输出尺度可能漂移。
- 专家没有收到 token 时仍要保持 DDP/compile 图稳定。
- aux loss 只在训练时有意义，推理时不应影响输出。

## 面试追问

::: details 参考回答：MoE 为什么需要辅助负载均衡损失？

router 如果只选少数专家，会造成热门专家过载、冷门专家无梯度，模型容量没有真正用起来。辅助损失给 router 一个均衡使用专家的训练信号。

:::

::: details 参考回答：MiniMind 的 MoE 与普通 FFN 的输出维度有什么关系？

MoE 最终仍要输出 hidden size，才能接回 Transformer 残差。区别只是中间计算由一个 FFN 变成按 token 路由到多个 expert 后再聚合。

:::

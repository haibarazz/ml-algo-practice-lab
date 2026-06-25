# MiniMind 系统依赖图：数据流、权重流和服务流笔记

## 关键公式与数据流

- 预训练边：`PretrainDataset -> MiniMindForCausalLM(loss) -> AdamW -> checkpoint`。
- SFT 边：`SFTDataset(label mask) -> same causal LM loss -> full_sft weight`。
- DPO 边：`DPODataset -> policy/ref logprob -> DPO loss -> dpo weight`。
- 推理边：`tokenizer prompt -> generate -> KV cache -> sampled tokens -> decode`。

## 易错点

- 把所有 trainer 都看成同一个训练脚本，会忽略 batch 结构和 loss 差异。
- 只画数据流不画权重流，会看不懂 `from_weight` 和阶段继承。
- 只画训练不画服务，会漏掉 tokenizer/template 在推理中的一致性问题。

## 面试追问

::: details 参考回答：MiniMind 的系统图里为什么要区分数据流和权重流？

数据流解释一个 batch 如何进入 loss，权重流解释不同训练阶段如何继承和保存模型。SFT、DPO、LoRA 的关键差别很多时候不在 forward，而在权重从哪里来、保存到哪里去。

:::

::: details 参考回答：为什么服务流也应该放进学习地图？

LLM 项目的目标不是只训练出 loss，而是能推理、对话或提供 API。服务流会暴露 tokenizer、chat template、KV cache、采样参数等与训练同样关键的工程约束。

:::

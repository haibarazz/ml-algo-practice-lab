# MiniMind 项目阶段地图笔记

## 关键公式与数据流

- 监督训练主线：`text -> tokenizer -> input_ids/labels -> model(input_ids, labels) -> loss -> optimizer.step()`。
- 偏好优化主线：`prompt -> chosen/rejected -> policy logprob/reference logprob -> preference loss -> optimizer.step()`。
- 推理主线：`prompt ids -> forward last token -> sample next token -> append -> update KV cache -> repeat`。

## 易错点

- 只按目录名读源码会漏掉跨目录依赖，例如 DPO 同时依赖 dataset、model、trainer_utils。
- 只看 README 不足以理解训练目标，关键逻辑通常在 dataset 和 trainer 中。

## 面试追问

::: details 参考回答：读一个开源 LLM 项目时，为什么要先画训练/推理阶段图？

阶段图能先回答“数据在哪里变形、loss 在哪里计算、权重在哪里保存、推理在哪里循环”。没有这张图，读者会陷入文件细节，看懂函数却不知道它服务哪条链路。

:::

::: details 参考回答：MiniMind 的学习顺序为什么不应该严格等于源码目录顺序？

源码目录按工程职责组织，学习顺序要按因果链组织。比如 DPO 目录上属于 trainer，但理解它必须先懂 DPODataset、causal LM logprob 和 reference model。

:::

# 预训练

理解 next-token loss、学习率调度、梯度累积、混合精度、checkpoint 和 DDP 训练骨架。

| 模块 | 学习重点 |
| --- | --- |
| [Causal LM Shift Loss：用当前位置预测下一个 token](./causal_lm_shift_loss/causal_lm_shift_loss.md) | 拆解 MiniMindForCausalLM 里 logits 与 labels 的错位交叉熵。 |
| [余弦学习率：从初始学习率平滑衰减](./cosine_lr_schedule/cosine_lr_schedule.md) | 复刻 `trainer_utils.get_lr` 中 MiniMind 使用的 cosine schedule。 |
| [梯度累积：用小 batch 模拟大 batch](./gradient_accumulation_counter/gradient_accumulation_counter.md) | 拆解 MiniMind 训练循环里 loss 缩放、反向传播和 optimizer step 的计数逻辑。 |

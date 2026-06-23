# 预训练

这个 section 按 MiniMind 学习路径组织，不按源码目录机械排序。

## Modules

- `causal_lm_shift_loss`: 复刻 MiniMindForCausalLM 中 labels 的 shift loss。
- `cosine_lr_schedule`: 复刻 trainer_utils.get_lr。
- `gradient_accumulation_counter`: 理解 train_epoch 中的 accumulation_steps。

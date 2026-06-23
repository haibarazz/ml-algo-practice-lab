# Tokenizer 与数据

这个 section 按 MiniMind 学习路径组织，不按源码目录机械排序。

## Modules

- `pretrain_token_pack`: 复刻 PretrainDataset 的 BOS/EOS/pad/label masking。
- `sft_assistant_label_mask`: 只让 assistant 回复段参与 SFT loss。
- `dpo_pair_shift`: 把 chosen/rejected 序列切成 x、y、mask。

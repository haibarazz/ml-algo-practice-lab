# Tokenizer 与数据

理解文本、对话和偏好样本怎样被整理成模型能训练的 token、label 和 mask。

| 模块 | 学习重点 |
| --- | --- |
| [预训练样本打包：BOS/EOS、pad 与 label mask](./pretrain_token_pack/pretrain_token_pack.md) | 复刻 `PretrainDataset` 怎样把一条普通文本变成 causal LM 训练样本。 |
| [SFT 标签掩码：只训练 assistant 回复](./sft_assistant_label_mask/sft_assistant_label_mask.md) | 拆解 `SFTDataset.generate_labels` 如何在多轮对话中只保留 assistant span 的监督信号。 |
| [DPO 偏好样本：chosen/rejected 的 x、y、mask](./dpo_pair_shift/dpo_pair_shift.md) | 复刻 `DPODataset` 怎样把一对偏好回答转换成 DPO 训练所需的 token 张量。 |

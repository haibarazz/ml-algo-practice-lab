# MiniMind Module Index

Source clone: `projects/minimind/source/` (ignored by this repo).

## Modules

### 00_project_map 项目地图

- `00_project_map/project_stage_map`: 从 MiniMind 文件名恢复学习阶段顺序。

### 01_tokenizer_and_data Tokenizer 与数据

- `01_tokenizer_and_data/pretrain_token_pack`: 复刻 PretrainDataset 的 BOS/EOS/pad/label masking。
- `01_tokenizer_and_data/sft_assistant_label_mask`: 只让 assistant 回复段参与 SFT loss。
- `01_tokenizer_and_data/dpo_pair_shift`: 把 chosen/rejected 序列切成 x、y、mask。

### 02_model_architecture 模型结构

- `02_model_architecture/rms_norm_minimind`: 手写 RMSNorm。
- `02_model_architecture/rope_rotate_half`: 复刻 MiniMind 的 rotate_half 形式 RoPE。
- `02_model_architecture/repeat_kv_for_gqa`: GQA/MQA 中把 KV heads 重复到 Q heads 数量。
- `02_model_architecture/swiglu_feed_forward`: 手写 MiniMind FFN 的门控结构。
- `02_model_architecture/moe_router_aux_loss`: MiniMind MoE 的 router 辅助损失。

### 03_pretraining 预训练

- `03_pretraining/causal_lm_shift_loss`: 复刻 MiniMindForCausalLM 中 labels 的 shift loss。
- `03_pretraining/cosine_lr_schedule`: 复刻 trainer_utils.get_lr。
- `03_pretraining/gradient_accumulation_counter`: 理解 train_epoch 中的 accumulation_steps。

### 04_sft SFT

- `04_sft/empty_think_cleanup`: 复刻 SFT/RLAIF 数据中的空 think 标签清理逻辑。

### 05_preference_alignment 偏好对齐

- `05_preference_alignment/dpo_sequence_loss`: 复刻 MiniMind 的序列级 DPO loss。
- `05_preference_alignment/per_token_logps`: 复刻 rollout_engine.compute_per_token_logps 的核心 gather。

### 06_inference 推理

- `06_inference/top_k_top_p_filter`: 手写 MiniMind generate 中的采样过滤。
- `06_inference/kv_cache_step_slice`: 理解 generate 中 past_key_values 对 input_ids 的切片。

### 07_evaluation 评估

- `07_evaluation/perplexity_from_losses`: 从 token 平均交叉熵估计困惑度。

### 08_system_architecture 系统架构

- `08_system_architecture/training_pipeline_edges`: 把 MiniMind 全链路拆成系统级有向图。

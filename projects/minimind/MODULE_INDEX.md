# MiniMind 模块索引

按源码学习路径排列。每个模块都包含源码导读、关键公式、手写练习、测试和参考答案。

## 项目地图

先建立 MiniMind 的全链路视角：源码目录、训练阶段、权重流和服务入口分别承担什么职责。

- [MiniMind 项目阶段地图](./00_project_map/project_stage_map/project_stage_map.md)：从源码文件名和训练脚本恢复 MiniMind 的学习顺序，先知道每一层在整条 LLM 流水线中的位置。

## Tokenizer 与数据

理解文本、对话和偏好样本怎样被整理成模型能训练的 token、label 和 mask。

- [预训练样本打包：BOS/EOS、pad 与 label mask](./01_tokenizer_and_data/pretrain_token_pack/pretrain_token_pack.md)：复刻 `PretrainDataset` 怎样把一条普通文本变成 causal LM 训练样本。
- [SFT 标签掩码：只训练 assistant 回复](./01_tokenizer_and_data/sft_assistant_label_mask/sft_assistant_label_mask.md)：拆解 `SFTDataset.generate_labels` 如何在多轮对话中只保留 assistant span 的监督信号。
- [DPO 偏好样本：chosen/rejected 的 x、y、mask](./01_tokenizer_and_data/dpo_pair_shift/dpo_pair_shift.md)：复刻 `DPODataset` 怎样把一对偏好回答转换成 DPO 训练所需的 token 张量。

## 模型结构

拆解 MiniMind Decoder-only Transformer 的归一化、位置编码、GQA、SwiGLU 与 MoE。

- [RMSNorm：只按均方根缩放的归一化](./02_model_architecture/rms_norm_minimind/rms_norm_minimind.md)：拆解 MiniMind 中 `RMSNorm` 的公式、数值稳定项和它在 Transformer block 中的位置。
- [RoPE 旋转位置编码：MiniMind 的 rotate_half 写法](./02_model_architecture/rope_rotate_half/rope_rotate_half.md)：从 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 理解 RoPE 怎样把位置信息注入 Q/K。
- [GQA/MQA：重复 KV head 以服务更多 Q head](./02_model_architecture/repeat_kv_for_gqa/repeat_kv_for_gqa.md)：拆解 MiniMind 的 `repeat_kv` 和 attention 中 `num_key_value_heads` 的显存/带宽意义。
- [SwiGLU FFN：门控前馈网络](./02_model_architecture/swiglu_feed_forward/swiglu_feed_forward.md)：拆解 MiniMind `FeedForward` 中 gate/up/down 三个投影如何组成现代 LLM 常见 FFN。
- [MoE 路由辅助损失：让专家负载更均衡](./02_model_architecture/moe_router_aux_loss/moe_router_aux_loss.md)：拆解 MiniMind `MOEFeedForward` 的 top-k 路由、专家聚合和 aux loss。

## 预训练

理解 next-token loss、学习率调度、梯度累积、混合精度、checkpoint 和 DDP 训练骨架。

- [Causal LM Shift Loss：用当前位置预测下一个 token](./03_pretraining/causal_lm_shift_loss/causal_lm_shift_loss.md)：拆解 MiniMindForCausalLM 里 logits 与 labels 的错位交叉熵。
- [余弦学习率：从初始学习率平滑衰减](./03_pretraining/cosine_lr_schedule/cosine_lr_schedule.md)：复刻 `trainer_utils.get_lr` 中 MiniMind 使用的 cosine schedule。
- [梯度累积：用小 batch 模拟大 batch](./03_pretraining/gradient_accumulation_counter/gradient_accumulation_counter.md)：拆解 MiniMind 训练循环里 loss 缩放、反向传播和 optimizer step 的计数逻辑。

## 监督微调

理解 chat template、assistant-only loss 和 thinking 标签清理怎样服务 SFT。

- [空 thinking 标签清理：控制推理风格和数据分布](./04_sft/empty_think_cleanup/empty_think_cleanup.md)：拆解 `post_processing_chat` 和 RLAIF/SFT 数据中 `<think>` 空块的随机保留逻辑。

## 偏好对齐

从 DPO 和 rollout logprob 入手，理解 chosen/rejected 偏好优化的核心张量。

- [DPO 序列级损失：让 chosen 相对 rejected 更可能](./05_preference_alignment/dpo_sequence_loss/dpo_sequence_loss.md)：拆解 MiniMind `train_dpo.py` 中 reference/policy logprob margin 的计算。
- [逐 token logprob：rollout 后如何评估生成概率](./05_preference_alignment/per_token_logps/per_token_logps.md)：复刻 `rollout_engine.compute_per_token_logps`，理解 RL/GRPO/PPO 中生成 token 的概率记录。

## 推理与采样

拆解 generate 中的 KV cache、temperature、top-k、top-p、重复惩罚和停止条件。

- [Top-k / Top-p 采样过滤：控制生成候选集合](./06_inference/top_k_top_p_filter/top_k_top_p_filter.md)：拆解 MiniMind `generate` 中 temperature、top-k、top-p 和 multinomial 的采样流程。
- [KV cache 增量推理：每步只算新 token](./06_inference/kv_cache_step_slice/kv_cache_step_slice.md)：拆解 MiniMind generate 中 `input_ids[:, past_len:]` 和 attention 中 past K/V 拼接。

## 评估

把交叉熵、困惑度和评测脚本的输出联系起来，理解语言模型基本评估量。

- [困惑度：从平均交叉熵到 PPL](./07_evaluation/perplexity_from_losses/perplexity_from_losses.md)：把语言模型 loss 转成更直观的 perplexity，并理解它的适用边界。

## 系统架构

把 tokenizer、数据集、模型、训练脚本、rollout 和服务脚本串成系统级依赖图。

- [MiniMind 系统依赖图：数据流、权重流和服务流](./08_system_architecture/training_pipeline_edges/training_pipeline_edges.md)：把 MiniMind 的源码节点串成可执行的系统架构图，理解训练产物如何进入推理服务。

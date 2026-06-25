# MiniMind 源码阅读指南

MiniMind 的学习价值不只是“有一个小 LLM 模型文件”，而是它把一个 LLM 项目从 tokenizer、数据集、模型结构、训练脚本、偏好优化、推理服务到评估都放在同一个较小代码库里。读它时要按链路读，而不是按目录逐个打开。

## 1. 数据先进入 tokenizer 和 Dataset

文本不会直接进入模型。`trainer/train_tokenizer.py` 展示 BPE tokenizer 的训练方式，`dataset/lm_dataset.py` 决定不同阶段的样本如何变成张量：

- 预训练样本：`text -> [BOS] + tokens + [EOS] + [PAD] -> labels`。
- SFT 样本：`conversations -> chat template -> input_ids -> assistant-only labels`。
- DPO 样本：`chosen/rejected -> x/y shift -> response mask`。
- RLAIF/Agent 样本：`prompt/messages/tools -> rollout -> reward 或规则判断`。

读这一层时要关注三个问题：哪些 token 参与 loss，哪些 token 只是条件，哪些位置必须被 mask。

## 2. 模型主干集中在 `model/model_minimind.py`

MiniMind 是 Decoder-only Causal LM。主干顺序是：

```text
input_ids
-> token embedding
-> N 个 MiniMindBlock
   -> RMSNorm
   -> GQA attention + RoPE + KV cache
   -> RMSNorm
   -> SwiGLU FFN 或 MoE FFN
-> final RMSNorm
-> LM Head
-> logits / loss / generate
```

这一层要重点读：

- `MiniMindConfig`：模型宽度、层数、head 数、KV head 数、RoPE、MoE 参数。
- `RMSNorm`：LLM 常用归一化，不依赖 batch。
- `precompute_freqs_cis` / `apply_rotary_pos_emb`：RoPE 位置编码。
- `repeat_kv` / `Attention`：GQA/MQA 和 KV cache 的核心 shape。
- `FeedForward` / `MOEFeedForward`：SwiGLU 与 MoE 路由。
- `MiniMindForCausalLM.forward`：next-token loss 的 shift。
- `generate`：temperature、top-k、top-p、repetition penalty、KV cache。

## 3. 训练脚本是一套共享骨架加不同数据/目标

`trainer/train_pretrain.py`、`trainer/train_full_sft.py`、`trainer/train_lora.py` 的外层循环非常相似：初始化分布式环境、构造模型和数据、设置混合精度、计算 loss、梯度累积、保存 checkpoint。差异主要来自 Dataset 和参数冻结方式。

核心训练公式是：

```text
loss = task_loss + aux_loss
loss = loss / accumulation_steps
backward
每 accumulation_steps 次执行 optimizer.step
```

MoE 打开时，`aux_loss` 来自专家路由均衡；不开 MoE 时它是 0。

## 4. 偏好优化从 token logprob 开始

DPO、PPO、GRPO 不是从“生成一句话好不好”直接更新模型，而是先把生成序列拆回 token 级 logprob：

```text
logits -> log_softmax -> gather(token_id) -> per-token logprob
```

DPO 比较 chosen/rejected 的序列 logprob margin；PPO/GRPO 比较新旧策略 logprob ratio，并结合 reward 或 advantage。`trainer/rollout_engine.py` 是连接 generate 和 RL loss 的关键文件。

## 5. 推理服务必须和训练模板一致

`eval_llm.py`、`scripts/web_demo.py`、`scripts/serve_openai_api.py` 都要依赖同一个 tokenizer 和 chat template。训练时 assistant span 怎么标记，推理时 prompt 就必须怎么组织；训练时使用哪些 special token，服务端也要一致。

推理性能的核心不是只看模型参数量，还要看：

- KV cache 的长度和 head 数。
- top-k/top-p/temperature 的采样策略。
- 是否使用流式输出。
- 是否把权重转换为 transformers 格式或服务端格式。

## 建议阅读顺序

1. `dataset/lm_dataset.py`
2. `model/model_minimind.py`
3. `trainer/trainer_utils.py`
4. `trainer/train_pretrain.py`
5. `trainer/train_full_sft.py`
6. `trainer/train_dpo.py`
7. `trainer/rollout_engine.py`
8. `model/model_lora.py`
9. `scripts/serve_openai_api.py` / `scripts/web_demo.py`
10. `eval_llm.py` / `scripts/eval_toolcall.py`

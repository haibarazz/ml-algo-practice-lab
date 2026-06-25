# MiniMind 源码映射表

本表按学习路径映射 MiniMind 源码。`projects/minimind/source/` 是本地源码副本，不会提交到仓库；公开文档只引用文件路径和关键行号。

| 学习阶段 | 主要源码 | 重点问题 |
| --- | --- | --- |
| 项目地图 | `README.md`, `README_en.md`, `trainer/*.py` | 训练阶段、权重继承、推理入口和发布格式 |
| Tokenizer 与数据 | `trainer/train_tokenizer.py`, `dataset/lm_dataset.py` | BPE、chat template、Pretrain/SFT/DPO/RLAIF/Agent 数据格式 |
| 模型结构 | `model/model_minimind.py`, `model/model_lora.py` | RMSNorm、RoPE、GQA、SwiGLU、MoE、LoRA、LM Head |
| 预训练 | `trainer/train_pretrain.py`, `trainer/trainer_utils.py` | Causal LM loss、学习率、梯度累积、混合精度、DDP、checkpoint |
| 监督微调 | `trainer/train_full_sft.py`, `trainer/train_lora.py`, `dataset/lm_dataset.py` | assistant-only labels、全参 SFT、LoRA 参数训练 |
| 偏好对齐 | `trainer/train_dpo.py`, `trainer/rollout_engine.py` | chosen/rejected、DPO margin、per-token logprob |
| RL 与 Agent | `trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py` | rollout、reward、advantage、tool call loop |
| 推理服务 | `model/model_minimind.py`, `scripts/serve_openai_api.py`, `scripts/web_demo.py`, `scripts/chat_api.py` | KV cache、采样过滤、OpenAI API、Web UI |
| 评估 | `eval_llm.py`, `scripts/eval_toolcall.py` | PPL、benchmark 调用、tool call 用例评估 |

## 模块覆盖范围

模块覆盖数据打包、SFT mask、DPO 数据、RMSNorm、RoPE、GQA、SwiGLU、MoE、causal LM loss、学习率、梯度累积、thinking 标签、DPO loss、per-token logprob、采样过滤、KV cache、PPL 和系统依赖图。

扩展阅读源码包括 `model/model_lora.py`、`trainer/train_lora.py`、`trainer/train_ppo.py`、`trainer/train_grpo.py`、`trainer/train_agent.py`、`scripts/serve_openai_api.py` 和 `scripts/convert_model.py`。

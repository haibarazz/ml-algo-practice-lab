# MiniMind Source Map

Local source clone: `projects/minimind/source/`
Upstream observed commit: `3f1a7cc` on `master` at clone time.

## Learning Path to Source Files

| 学习阶段 | 主要源码 | 关键点 |
| --- | --- | --- |
| Project map | `README.md`, `README_en.md` | 项目能力边界、训练阶段和发布模型说明 |
| Tokenizer/Data | `trainer/train_tokenizer.py`, `dataset/lm_dataset.py` | tokenizer 训练、Pretrain/SFT/DPO/RLAIF/Agent 数据格式 |
| Model | `model/model_minimind.py`, `model/model_lora.py` | Config、RMSNorm、RoPE、GQA、SwiGLU、MoE、LM Head、generate |
| Pretrain | `trainer/train_pretrain.py`, `trainer/trainer_utils.py` | Causal LM loss、梯度累积、checkpoint、DDP |
| SFT/LoRA | `trainer/train_full_sft.py`, `trainer/train_lora.py` | assistant-only labels、全参 SFT、LoRA 参数训练 |
| Preference/RL | `trainer/train_dpo.py`, `trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py`, `trainer/rollout_engine.py` | DPO、PPO/GRPO、rollout、reward、tool/agent loop |
| Inference | `model/model_minimind.py`, `scripts/serve_openai_api.py`, `scripts/web_demo.py`, `scripts/chat_api.py` | KV cache、top-k/top-p、OpenAI API、Web UI |
| Evaluation | `eval_llm.py`, `scripts/eval_toolcall.py` | benchmark 调用、tool call 用例评估 |
| System architecture | `trainer/*.py`, `dataset/*.py`, `scripts/*.py` | 数据流、权重流、服务流和训练阶段依赖 |

## Current Dissection Coverage

See `MODULE_INDEX.md`. Each module keeps its own `Source Mapping` section and uses a small hand-write exercise to isolate the source mechanism.

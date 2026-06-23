# Training Pipeline Edges Notes

## Source Mapping

- `README.md:82-94`
- `trainer/train_tokenizer.py:24-166`
- `trainer/train_pretrain.py:83-171`
- `trainer/train_full_sft.py:84-172`
- `trainer/train_dpo.py:131-227`
- `trainer/rollout_engine.py:23-92`
- `scripts/serve_openai_api.py:28-245`

## 常见坑

- 系统图要能解释权重从哪里来、数据在哪里变形。
- RL/Agentic RL 可以作为 DPO 后的扩展训练分支。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

# MoE Router Aux Loss Notes

## Source Mapping

- `model/model_minimind.py:148-176`

## 常见坑

- MiniMind 默认 num_experts_per_tok=1，因此 load shape 通常是 `[1, E]`。
- 如果 top-k>1，要确认归一化口径。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

# Per Token Log Probs Notes

## Source Mapping

- `trainer/rollout_engine.py:23-36`

## 常见坑

- logits 和 token_ids 的 seq_len 必须一致。
- 真实代码会配合 logits_to_keep 减少计算。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

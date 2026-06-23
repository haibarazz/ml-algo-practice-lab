# Top-k Top-p Filter Notes

## Source Mapping

- `model/model_minimind.py:257-288`

## 常见坑

- MiniMind 的实现会把超过 top-p 的 mask 右移，保证第一个 token 保留。
- top_k 和 top_p 同时开时，top_k 先执行。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

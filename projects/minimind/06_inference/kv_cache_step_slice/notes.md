# KV Cache Step Slice Notes

## Source Mapping

- `model/model_minimind.py:257-288`
- `model/model_minimind.py:120-124`
- `model/model_minimind.py:209-232`

## 常见坑

- KV cache 不是缓存 logits，而是每层 attention 的 K/V。
- attention_mask 仍要随生成长度增长。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

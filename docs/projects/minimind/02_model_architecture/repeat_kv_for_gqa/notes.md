# Repeat KV for GQA Notes

## Source Mapping

- `model/model_minimind.py:86-89`
- `model/model_minimind.py:91-134`

## 常见坑

- 这里重复的是 KV，不是 Q。
- 真实 attention 前还会 transpose 到 `[B, H, S, D]`。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

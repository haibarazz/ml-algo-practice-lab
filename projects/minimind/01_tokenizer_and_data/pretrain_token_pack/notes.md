# Pretrain Token Pack Notes

## Source Mapping

- `dataset/lm_dataset.py:37-55`

## 常见坑

- 注意 max_length 必须至少容纳 BOS/EOS。
- labels 忽略 pad，input_ids 仍保留 pad。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

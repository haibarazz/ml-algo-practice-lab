# DPO Pair Shift Notes

## Source Mapping

- `dataset/lm_dataset.py:122-192`

## 常见坑

- mask 和 y 对齐，不是和 x 对齐。
- chosen/rejected 顺序会影响 DPO loss 的符号。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

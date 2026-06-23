# Gradient Accumulation Counter Notes

## Source Mapping

- `trainer/train_pretrain.py:24-80`
- `trainer/train_full_sft.py:24-80`

## 常见坑

- 真实训练中 backward 每个 batch 都做，optimizer step 不是每个 batch 都做。
- loss 除以 accumulation_steps 是为了保持梯度尺度。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

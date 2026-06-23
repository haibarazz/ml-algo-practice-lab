# Causal LM Shift Loss Notes

## Source Mapping

- `model/model_minimind.py:245-253`
- `trainer/train_pretrain.py:24-80`

## 常见坑

- logits 和 labels 的错位是最常见 bug。
- ignore_index 是为了屏蔽 pad 或非 assistant token。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

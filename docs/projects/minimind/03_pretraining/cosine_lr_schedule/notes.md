# Cosine LR Schedule Notes

## Source Mapping

- `trainer/trainer_utils.py:40-41`

## 常见坑

- 没有 warmup；这是项目当前实现，不是通用最佳实践。
- total_steps 不能为 0。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

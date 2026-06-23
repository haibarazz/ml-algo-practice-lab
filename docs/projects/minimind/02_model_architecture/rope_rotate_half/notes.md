# RoPE Rotate Half Notes

## Source Mapping

- `model/model_minimind.py:62-84`

## 常见坑

- 有些实现按偶奇维成对旋转，MiniMind 这里是 half split 形式。
- cos/sin 来自预计算 position embedding。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

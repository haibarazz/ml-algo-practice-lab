# Empty Think Cleanup Notes

## Source Mapping

- `dataset/lm_dataset.py:31-35`
- `dataset/lm_dataset.py:106-119`
- `dataset/lm_dataset.py:195-224`

## 常见坑

- 真实代码里 keep/remove 是随机比例控制。
- 这是数据增强/模板鲁棒性的一部分。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

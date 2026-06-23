# SFT Assistant Label Mask Notes

## Source Mapping

- `dataset/lm_dataset.py:58-119`

## 常见坑

- 真实项目中的 assistant_bos/eos 来自 tokenizer 编码。
- 如果 eos 被截断，labels 会延伸到 max_length。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

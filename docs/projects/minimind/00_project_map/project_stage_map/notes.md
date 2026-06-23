# Project Stage Map Notes

## Source Mapping

- `README.md:82-94`
- `model/model_minimind.py:10-288`
- `dataset/lm_dataset.py:37-252`
- `trainer/train_pretrain.py:83-171`
- `trainer/train_full_sft.py:84-172`
- `trainer/train_dpo.py:25-50`
- `scripts/serve_openai_api.py:28-245`

## 常见坑

- 面试追问：为什么不按目录顺序学？
- 系统设计追问：source map 为什么要保留？

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

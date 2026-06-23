# Perplexity from Losses Notes

## Source Mapping

- `eval_llm.py:12-93`
- `model/model_minimind.py:245-253`

## 常见坑

- 不能简单平均 batch loss，除非每个 batch token 数一样。
- perplexity 越低通常表示语言建模越好。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

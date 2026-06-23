# SwiGLU Feed Forward Notes

## Source Mapping

- `model/model_minimind.py:136-146`

## 常见坑

- 如果 hidden_act 换掉，gate 分支的激活也会变。
- 三个矩阵 shape 要能完成 hidden -> intermediate -> hidden。

## 可继续追问

- 这个最小实现和 MiniMind 源码中的真实张量 shape 有什么差别？
- 如果 batch size、seq len、hidden size 变大，哪里会先成为瓶颈？
- 这个模块在 Pretrain / SFT / DPO / Inference 哪个阶段最容易出错？

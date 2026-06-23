# Dropout Notes

## 核心公式

- inverted dropout 训练态：$y=\frac{m\odot x}{1-p}$，$m\sim Bernoulli(1-p)$。
- 推理态通常直接 $y=x$。

## 易错点

- 推理时还随机丢弃，导致输出不稳定。
- 训练时没有除以 keep probability，导致期望变小。
- `p` 和 `keep_prob` 混淆。
- mask 需要和输入 shape 可广播，并且通常只在训练态采样。

## 面试追问

- Dropout 为什么能缓解过拟合？
- 什么是 inverted dropout？为什么推理时不用再缩放？
- Dropout 和 BatchNorm 同时使用有什么注意点？
- Transformer 中 attention dropout 和 FFN dropout 分别作用在哪里？

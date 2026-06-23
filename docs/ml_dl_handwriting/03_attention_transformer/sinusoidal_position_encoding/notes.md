# Sinusoidal Position Encoding Notes

## 核心公式

- $PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}})$。
- $PE_{pos,2i+1}=\cos(pos/10000^{2i/d_{model}})$。

## 易错点

- 指数写成 `i/d` 还是 `2i/d` 混乱，本实现按偶数维索引配对。
- 奇数 `d_model` 时 cos 维度切片要对齐。
- 位置编码要和 token embedding shape 对齐后相加。
- 绝对位置编码外推长度时不一定和训练分布一致。

## 面试追问

- Transformer 为什么需要位置编码？
- 正弦位置编码为什么使用不同频率？
- 绝对位置编码、可学习位置编码、RoPE 有什么区别？
- 正弦位置编码为什么被认为具备一定长度外推能力？

# Sinusoidal Position Encoding Notes

## 易错点
- 指数写成 `i/d` 还是 `2i/d` 混乱，本实现用偶数索引数组。
- 奇数 d_model 时 cos 维度切片要对齐。

## 面试追问
- 绝对位置编码和 RoPE 区别？
- 为什么使用不同频率？

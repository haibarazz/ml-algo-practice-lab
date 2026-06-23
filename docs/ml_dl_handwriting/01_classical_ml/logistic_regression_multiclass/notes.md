# Multiclass Logistic Regression Notes

## 易错点
- 忘记稳定 softmax。
- `grad = probs` 后原地修改，后续再用 probs 会变脏。
- W shape 写成 `[C,D]` 导致乘法方向混乱。

## 面试追问
- softmax regression 和 one-vs-rest 区别？
- CE 梯度为什么可以化简？

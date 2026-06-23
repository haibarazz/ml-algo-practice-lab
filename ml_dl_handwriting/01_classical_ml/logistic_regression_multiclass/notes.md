# Multiclass Logistic Regression Notes

## 核心公式

- $p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}$，$z=XW+b$。
- $L=-\log p_{y}$，对 logits 的梯度为 $p-\operatorname{onehot}(y)$。

## 易错点

- 忘记稳定 softmax，logits 较大时容易 overflow。
- `grad = probs` 后原地修改，后续再用 `probs` 会被污染。
- W shape 写成 `[C,D]` 或 `[D,C]` 时要全程保持一致。
- one-vs-rest 和 multinomial softmax 的训练目标不同，不要混用结论。

## 面试追问

- softmax regression 和 one-vs-rest logistic regression 有什么区别？
- cross entropy 梯度为什么能化简为 `probs - onehot`？
- 多分类类别不均衡时可以怎样加权？
- softmax 输出概率是否一定校准？如果不校准怎么办？

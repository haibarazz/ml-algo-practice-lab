# Softmax Stable Notes

## 核心公式

- $\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$。
- 数值稳定写法：$\operatorname{softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}$。

## 易错点

- 忘记减最大值，遇到大 logits 会 overflow。
- `axis` 写死成最后一维，批量输入时不够通用。
- 没有 `keepdims=True`，二维或高维输入广播容易错。
- softmax 输出不是独立概率；所有类别概率相加必须为 1。

## 面试追问

::: details 参考回答：为什么减去同一个常数不会改变 softmax 结果？

softmax 的分子分母都会同时乘上 `exp(-c)`，这个公共因子会被约掉，所以减去同一个常数不改变结果。通常取最大值作为 `c`，可以让最大的指数项变成 1，避免 `exp` 上溢。

:::

::: details 参考回答：softmax 和 sigmoid 在二分类、多分类场景下有什么关系？

二分类时，两个 logits 的 softmax 可以化成对 logit 差值做 sigmoid，所以 sigmoid 可以看成二分类 softmax 的特例。多分类单标签场景用 softmax，因为类别之间互斥；多标签场景通常每个标签独立用 sigmoid。

:::

::: details 参考回答：为什么交叉熵里常把 log-softmax 和 NLL 合并计算？

分开算 softmax 再取 log 容易遇到上溢、下溢和 `log(0)`。把 log-softmax 和 NLL 合并后可以用 log-sum-exp 稳定计算，同时避免保存不必要的中间概率。

:::

::: details 参考回答：temperature 对 softmax 分布的尖锐程度有什么影响？

temperature 越小，logits 被放大，分布越尖锐，模型更偏向最大 logit。temperature 越大，分布越平滑，常用于蒸馏、采样多样性控制和校准分析。

:::

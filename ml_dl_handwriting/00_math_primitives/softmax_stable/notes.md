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

- 为什么减去同一个常数不会改变 softmax 结果？
- softmax 和 sigmoid 在二分类、多分类场景下有什么关系？
- 为什么交叉熵里常把 log-softmax 和 NLL 合并计算？
- temperature 对 softmax 分布的尖锐程度有什么影响？

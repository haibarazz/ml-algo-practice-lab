# Softmax Stable Notes

## 易错点

- 忘记减最大值，遇到大 logits 会 overflow。
- `axis` 写死成最后一维，batch 输入时不够通用。
- 没有 `keepdims=True`，二维输入广播容易错。

## 面试追问

- 为什么减最大值不改变 softmax？
- softmax 和 sigmoid 的关系是什么？
- cross entropy 里为什么常用 log-softmax 合并计算？

# Causal LM Shift Loss：用当前位置预测下一个 token笔记

## 关键公式与数据流

- $L=-\frac{1}{N}\sum_{t\in valid}\log p_\theta(x_{t+1}|x_{\le t})$。
- $x=logits_{0:T-1}$，$y=labels_{1:T}$。
- 若 $labels_t=-100$，该位置不计入 $N$ 和 loss。

## 易错点

- 不 shift 会变成预测当前 token，模型能作弊。
- shift 两次会让监督错位。
- flatten 前后 batch/seq 顺序必须一致。

## 面试追问

::: details 参考回答：为什么 causal LM 要 shift 一位？

位置 t 的 hidden state 只能看见 t 及之前的 token，所以它的监督目标应该是下一个 token。shift 一位正是把输入上下文和下一个 token 标签对齐。

:::

::: details 参考回答：`ignore_index=-100` 在预训练和 SFT 中分别屏蔽什么？

预训练中主要屏蔽 padding；SFT 中还屏蔽 system/user/prompt 等非 assistant 位置。两者都通过同一个 cross entropy 参数生效。

:::

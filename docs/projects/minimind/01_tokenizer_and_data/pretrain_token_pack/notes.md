# 预训练样本打包：BOS/EOS、pad 与 label mask笔记

## 关键公式与数据流

- 打包：$ids=[BOS] + tokenizer(text)[:L-2] + [EOS] + [PAD]\times(L-|ids|)$。
- 标签：$labels_i=ids_i$，若 $ids_i=PAD$，则 $labels_i=-100$。
- 模型内 shift：用 $logits_{t}$ 预测 $labels_{t+1}$。

## 易错点

- 忘记给 BOS/EOS 留位置，会导致末尾 EOS 被截断。
- labels 不屏蔽 pad，会让模型学习预测 padding。
- 在 dataset 和 model 两边重复 shift，会产生错位两格的隐蔽错误。

## 面试追问

::: details 参考回答：为什么预训练时 `input_ids` 和 `labels` 可以先设成一样？

因为 causal LM 的错位预测在模型 loss 中完成：`logits[..., :-1, :]` 对齐 `labels[..., 1:]`。数据集只需要提供完整 token 序列和无效位置 mask。

:::

::: details 参考回答：为什么 pad label 用 `-100` 而不是 pad token id？

`torch.nn.functional.cross_entropy` 默认用 `ignore_index=-100` 跳过这些位置。若使用 pad token id，pad 会成为一个真实监督目标，长短样本的 padding 数量还会改变 loss 权重。

:::

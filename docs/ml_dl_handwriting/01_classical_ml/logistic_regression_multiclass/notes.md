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

::: details 参考回答：softmax regression 和 one-vs-rest logistic regression 有什么区别？

softmax regression 一次性建模互斥多分类分布，所有类别概率共同归一化。one-vs-rest 会训练多个二分类器，类别之间不天然竞争，输出概率也不一定相互校准。

:::

::: details 参考回答：cross entropy 梯度为什么能化简为 `probs - onehot`？

softmax + cross entropy 对 logits 的梯度会化简为 `probs - onehot`。直觉上，模型给某类的概率高于目标就降低它，目标类概率低就提高它。

:::

::: details 参考回答：多分类类别不均衡时可以怎样加权？

可以在 cross entropy 中加入 class weight，让少数类或高成本类别的错误更重要。也可以重采样、使用 focal loss，或按业务指标调阈值，但要用验证集确认没有牺牲关键指标。

:::

::: details 参考回答：softmax 输出概率是否一定校准？如果不校准怎么办？

softmax 输出不一定校准，它只是相对分数归一化后的结果，深度模型常会过度自信。可以用 temperature scaling、Platt scaling、isotonic regression 或校准集后处理。

:::

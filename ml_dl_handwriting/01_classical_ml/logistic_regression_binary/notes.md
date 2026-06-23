# Binary Logistic Regression Notes

## 核心公式

- $p(y=1|x)=\sigma(w^\top x+b)$，$\sigma(z)=\frac{1}{1+e^{-z}}$。
- $L=-[y\log p+(1-y)\log(1-p)]$，对 logits 的梯度为 $p-y$。

## 易错点

- 把 logistic regression 和线性回归混淆；LR 输出的是类别概率。
- sigmoid 直接计算可能溢出，BCE with logits 更稳定。
- BCE 梯度符号写反，参数更新方向会错。
- 默认阈值 0.5 不一定适合类别不均衡或不同业务成本。

## 面试追问

::: details 参考回答：逻辑回归为什么是判别模型？

逻辑回归直接建模 `P(y|x)`，学习的是给定特征下类别的条件概率，而不是建模 `P(x|y)` 和 `P(y)`。因此它属于判别模型，关注决策边界而不是数据生成过程。

:::

::: details 参考回答：逻辑回归和线性回归的目标函数有什么区别？

线性回归通常最小化连续值预测的平方误差，输出不受概率范围约束。逻辑回归用 sigmoid 把线性打分映射成概率，并用 BCE 或极大似然训练分类边界。

:::

::: details 参考回答：为什么 BCE with logits 比 sigmoid 后再 BCE 更稳定？

`BCE with logits` 可以把 sigmoid 和 log loss 合并成稳定公式，避免大正数或大负数 logits 下的溢出和 `log(0)`。分开算 sigmoid 后再 BCE，极端概率会让数值精度和梯度都更脆弱。

:::

::: details 参考回答：二分类阈值如何根据 precision/recall 或业务成本调整？

阈值应该根据验证集上的 precision、recall、F1、PR 曲线或业务成本选择。比如漏报成本高就降低阈值提高 recall，误报成本高就提高阈值改善 precision。

:::

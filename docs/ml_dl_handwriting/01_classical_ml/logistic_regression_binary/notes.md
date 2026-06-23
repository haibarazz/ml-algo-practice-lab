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

- 逻辑回归为什么是判别模型？
- 逻辑回归和线性回归的目标函数有什么区别？
- 为什么 BCE with logits 比 sigmoid 后再 BCE 更稳定？
- 二分类阈值如何根据 precision/recall 或业务成本调整？

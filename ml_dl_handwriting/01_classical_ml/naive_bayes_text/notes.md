# Naive Bayes Text Notes

## 核心公式

- $\hat y=\arg\max_c \log P(c)+\sum_j x_j\log P(w_j|c)$。
- Laplace 平滑：$P(w|c)=\frac{count(w,c)+\alpha}{\sum_v count(v,c)+\alpha |V|}$。

## 易错点

- 概率直接相乘会下溢，应在 log 空间相加。
- 未见词概率为 0，需要平滑。
- 类别先验忘记加入，类别不均衡时影响明显。
- 多项式 NB、伯努利 NB、Gaussian NB 的特征假设不同。

## 面试追问

- 朴素贝叶斯“朴素”在哪里？独立性假设为什么仍常有效？
- 多项式 NB 和伯努利 NB 的区别是什么？
- 为什么文本分类里常在 log 空间计算？
- 平滑系数 alpha 太大或太小会产生什么影响？

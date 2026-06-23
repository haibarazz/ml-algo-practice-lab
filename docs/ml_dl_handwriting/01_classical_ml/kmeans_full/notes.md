# KMeans Full Notes

## 核心公式

- 目标函数：$\min_{\{C_k\},\{\mu_k\}}\sum_k\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2$。
- 每轮 assignment/update 不增加该目标，但只能保证收敛到局部最优。

## 易错点

- 收敛判断放错位置，导致多跑一轮或提前停止。
- 空簇导致 `mean of empty slice`，必须定义处理策略。
- 初始化不同结果不同，工程中常跑多次取 inertia 最小的结果。
- 没有设置 `max_iter`，异常数据上可能运行过久。

## 面试追问

- KMeans 为什么只能保证局部最优？
- KMeans++ 的初始化思想是什么？
- KMeans 适合非凸簇或不同密度簇吗？为什么？
- KMeans、GMM、DBSCAN 的核心假设有什么不同？

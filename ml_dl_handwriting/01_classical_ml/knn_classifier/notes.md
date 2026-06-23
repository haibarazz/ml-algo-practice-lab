# KNN Classifier Notes

## 核心公式

- $\hat y=\operatorname{mode}\{y_i: x_i \in N_k(x)\}$。
- 常见距离：欧氏距离 $\lVert x-x_i\rVert_2$，也可按任务换成曼哈顿距离或余弦距离。

## 易错点

- `k` 大于训练样本数没有处理。
- 投票 tie-break 不确定，复现实验时必须固定规则。
- 特征尺度不同会显著影响距离，通常要做标准化。
- KNN 训练很轻，但预测要扫训练集，线上延迟可能很高。

## 面试追问

- KNN 的训练复杂度和预测复杂度分别是多少？
- 如何选择 K？K 太大或太小分别有什么问题？
- 如何加速最近邻检索，比如 KDTree、BallTree、ANN？
- KNN 为什么容易受维度灾难影响？

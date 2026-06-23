# Linear Regression GD Notes

## 核心公式

- $\hat y=Xw+b$，$L=\frac{1}{n}\sum_i(\hat y_i-y_i)^2$。
- $\nabla_w L=\frac{2}{n}X^\top(Xw+b-y)$，$\nabla_b L=\frac{2}{n}\sum_i(\hat y_i-y_i)$。

## 易错点

- 梯度忘记除以 batch size，导致学习率含义变化。
- bias 梯度写成向量而不是标量或按输出维聚合。
- 学习率过大导致发散，过小导致收敛慢。
- 没有检查输入是否需要加截距项，闭式解和 GD 实现容易不一致。

## 面试追问

- 线性回归的闭式解是什么？什么时候不适合直接用闭式解？
- GD、SGD、mini-batch SGD 的区别是什么？
- 为什么特征标准化会影响梯度下降收敛速度？
- L1/L2 正则会怎样改变目标函数和解的性质？

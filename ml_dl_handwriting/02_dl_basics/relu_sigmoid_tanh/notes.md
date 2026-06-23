# Activation Notes

## 核心公式

- ReLU：$f(x)=\max(0,x)$；Sigmoid：$\sigma(x)=1/(1+e^{-x})$；Tanh：$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$。
- 导数：$\sigma'(x)=\sigma(x)(1-\sigma(x))$，$\tanh'(x)=1-\tanh^2(x)$。

## 易错点

- sigmoid 直接 `exp(-x)` 在大负数上可能溢出。
- ReLU 在 0 处不可导，需要说明实现约定。
- tanh 梯度公式应是 `1 - tanh(x)^2`，不是 `1 - x^2`。
- sigmoid/tanh 饱和区梯度很小，深层网络中容易梯度消失。

## 面试追问

- ReLU 为什么能缓解梯度消失？它有什么缺点？
- sigmoid、tanh、ReLU 的输出范围和梯度特点分别是什么？
- 什么是 dead ReLU？如何缓解？
- GeLU/SiLU 相比 ReLU 的直觉优势是什么？

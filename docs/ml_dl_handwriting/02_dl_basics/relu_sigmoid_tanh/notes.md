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

::: details 参考回答：ReLU 为什么能缓解梯度消失？它有什么缺点？

ReLU 在正区间导数为 1，不像 sigmoid/tanh 在大幅度输入时导数接近 0，所以能缓解饱和导致的梯度消失。缺点是负区间梯度为 0，学习率过大或偏置不当时神经元可能死亡。

:::

::: details 参考回答：sigmoid、tanh、ReLU 的输出范围和梯度特点分别是什么？

sigmoid 输出在 0 到 1，适合概率门控，但两端饱和且非零中心。tanh 输出在 -1 到 1，零中心但仍会饱和；ReLU 输出非负，计算简单，正区间梯度稳定，但可能 dead ReLU。

:::

::: details 参考回答：什么是 dead ReLU？如何缓解？

dead ReLU 指神经元长期落在负半轴，输出和梯度都为 0，几乎不再更新。可以用较小学习率、合适初始化、Leaky ReLU、PReLU 或调整归一化来缓解。

:::

::: details 参考回答：GeLU/SiLU 相比 ReLU 的直觉优势是什么？

GELU 和 SiLU 都是平滑激活，允许输入按大小连续地通过，而不是像 ReLU 一刀切。它们在 Transformer/LLM 中常表现更好，直觉上提供了更柔和的数据依赖门控。

:::

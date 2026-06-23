# FFN Notes

## 核心公式

- Transformer FFN 常见形式：$FFN(x)=W_2\phi(W_1x+b_1)+b_2$。
- GELU 常用近似：$GELU(x)\approx0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$。

## 易错点

- 把 FFN 和 attention 混在一起；FFN 是逐 token 的非线性变换。
- 忘记第二个线性层，输出维度无法回到 `d_model`。
- GeLU 公式写错，或把输入当成 sigmoid gate。
- FFN expansion ratio 会显著影响参数量和计算量。

## 面试追问

::: details 参考回答：Transformer 中 FFN 的参数量通常占比如何？

Transformer 中 FFN 往往占很大参数比例，因为它包含从 `d_model` 到更大 hidden size 再回到 `d_model` 的两层线性变换。常见 expansion ratio 为 4 时，FFN 参数量通常超过 attention 投影。

:::

::: details 参考回答：FFN 为什么可以逐 token 并行计算？

FFN 对每个 token 独立应用同一组线性层和激活，不依赖其他 token。token 间信息交换已经由 attention 完成，所以 FFN 可以在序列维上完全并行。

:::

::: details 参考回答：GELU 相比 ReLU 的直觉优势是什么？

GELU 是平滑的概率式门控，输入越大通过越多，输入较小时被柔和抑制。相比 ReLU 的硬截断，它在 Transformer 中常带来更平滑的优化和更好的表达。

:::

::: details 参考回答：FFN hidden size 为什么常设为 `4 * d_model` 或相近比例？

`4 * d_model` 是经验上兼顾容量和计算的宽度设置，给逐 token 非线性变换足够的中间维度。现代 LLM 使用 SwiGLU 等门控 FFN 时，hidden size 常会调整到约 `8/3 * d_model` 以控制参数量。

:::

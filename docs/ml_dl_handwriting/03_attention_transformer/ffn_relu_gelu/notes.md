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

- Transformer 中 FFN 的参数量通常占比如何？
- FFN 为什么可以逐 token 并行计算？
- GELU 相比 ReLU 的直觉优势是什么？
- FFN hidden size 为什么常设为 `4 * d_model` 或相近比例？

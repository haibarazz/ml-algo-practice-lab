# GQA/MQA：重复 KV head 以服务更多 Q head

拆解 MiniMind 的 `repeat_kv` 和 attention 中 `num_key_value_heads` 的显存/带宽意义。

## 学习目标

- 区分 MHA、MQA、GQA 的 head 数关系。
- 理解为什么 KV cache 大小由 KV head 数决定。
- 掌握 `repeat_kv` 在 head 维重复而不是 seq 维重复。

## MiniMind 源码定位

- `model/model_minimind.py:86-89`
- `model/model_minimind.py:91-134`

## 源码机制详解

MiniMind 配置里 `num_attention_heads` 默认是 8，`num_key_value_heads` 默认是 4，因此每两个 Q head 共享一组 K/V。这就是 GQA：Query head 多，Key/Value head 少。
`repeat_kv` 接收 `[batch, seq, kv_heads, head_dim]`，先插入一个 repeat 维，再 expand 到 `n_rep`，最后 reshape 成 `[batch, seq, q_heads, head_dim]`。它是在注意力计算前把共享 K/V 展开成与 Q head 数一致的形状。
推理时 KV cache 保存每层每个历史 token 的 K/V。减少 KV head 数会近似线性降低 cache 显存和读取带宽，所以 GQA 是大模型推理优化中非常关键的结构折中。

## 关键公式与数据流

- MHA：$H_q=H_k=H_v$；MQA：$H_k=H_v=1$；GQA：$1 < H_{kv} < H_q$。
- $n_{rep}=H_q/H_{kv}$。
- KV cache 规模近似正比于 $layers \times seq \times H_{kv}\times head\_dim$。

## 为什么练这个

- 手写 repeat_kv 能直接理解 GQA 的 shape 变化。
- 这也是推理显存优化和 KV cache 章节的前置知识。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def repeat_kv(x, n_rep):
    """带提示实现。"""
    # TODO 1: n_rep 为 1 时直接返回
    # TODO 2: 在 head 维后插入 repeat 维
    # TODO 3: reshape 成重复后的 head 数量
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def repeat_kv(x, n_rep):
    """无提示实现。"""
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现无提示函数后直接运行测试区代码。

```python
import numpy as np


def test_repeat_kv():
    x = np.array([[[[1], [2]], [[3], [4]]]])
    out = repeat_kv(x, 2)
    assert out.shape == (1, 2, 4, 1)
    assert out[0, 0, :, 0].tolist() == [1, 1, 2, 2]
    assert repeat_kv(x, 1).shape == x.shape


test_repeat_kv()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def repeat_kv(x, n_rep):
    x = np.asarray(x)
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return np.repeat(x[:, :, :, None, :], n_rep, axis=3).reshape(b, s, h * n_rep, d)
```

### 解析

1. 这里重复的是 KV，不是 Q。
2. 真实 attention 前还会 transpose 到 `[B, H, S, D]`。

## 工程要点 / 面试追问

### 关键公式与数据流

- MHA：$H_q=H_k=H_v$；MQA：$H_k=H_v=1$；GQA：$1 < H_{kv} < H_q$。
- $n_{rep}=H_q/H_{kv}$。
- KV cache 规模近似正比于 $layers \times seq \times H_{kv}\times head\_dim$。

### 易错点

- 在 seq 维 repeat 会复制时间步，语义完全错。
- `num_attention_heads` 必须能被 `num_key_value_heads` 整除。
- 只看参数量会低估 GQA 价值，它主要优化推理 cache 和带宽。

### 面试追问

::: details 参考回答：GQA 相比 MHA 主要省在哪里？

主要省 KV cache 和推理时读取 K/V 的带宽。Q head 仍然多，但 K/V head 更少，历史 token 的缓存规模随 KV head 数下降。

:::

::: details 参考回答：`repeat_kv` 和真的学习更多 KV head 有什么区别？

`repeat_kv` 只是把少量已学习的 K/V 表示广播给多个 Q head，不增加 K/V 参数和 cache。学习更多 KV head 会增加表达多样性，但也增加显存和带宽成本。

:::

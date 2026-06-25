# RoPE 旋转位置编码：MiniMind 的 rotate_half 写法

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/02_model_architecture/rope_rotate_half/rope_rotate_half.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


从 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 理解 RoPE 怎样把位置信息注入 Q/K。

## 学习目标

- 理解 MiniMind 为什么只对 Q/K 应用 RoPE。
- 掌握 rotate_half 形式和 cos/sin 广播。
- 理解 KV cache 推理时 position offset 为什么必须连续。

## MiniMind 源码定位

- `model/model_minimind.py:62-84`
- `model/model_minimind.py:111-124`

## 源码机制详解

`precompute_freqs_cis` 为每个位置和 head 维度提前生成 cos/sin 表。MiniMind 把半维 cos/sin 复制成完整 head_dim，配合 `rotate_half` 完成二维子空间旋转。
`apply_rotary_pos_emb` 的核心是 `q*cos + rotate_half(q)*sin` 和同样的 k 变换。这样 Q/K 点积会携带相对位置信息，而 V 不需要旋转，因为 V 只承载被加权汇聚的内容。
在 `MiniMindModel.forward` 中，位置表切片从 `start_pos` 开始。`start_pos` 来自 past KV 长度，保证增量推理时新 token 的位置接在历史 token 后面，而不是每一步都从 0 开始。

## 关键公式与数据流

- 二维旋转：$(x_1,x_2)\mapsto(x_1\cos\theta-x_2\sin\theta,\ x_1\sin\theta+x_2\cos\theta)$。
- MiniMind 写法：$rope(x)=x\odot cos + rotate\_half(x)\odot sin$。
- RoPE 作用于 Q/K，使 attention score 依赖相对位置差。

## 为什么练这个

- 手写 rotate_half 可以把 RoPE 从抽象公式落到张量维度操作。
- 这个练习也服务后续 KV cache，因为位置 offset 是推理正确性的核心。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def apply_rope_rotate_half(x, cos, sin):
    """带提示实现。"""
    # TODO 1: 实现 rotate_half
    # TODO 2: 确保 cos/sin 可广播
    # TODO 3: 返回 x*cos + rotate_half(x)*sin
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def apply_rope_rotate_half(x, cos, sin):
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


def test_apply_rope_rotate_half():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    cos = np.ones(4)
    sin = np.zeros(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), x)
    cos = np.zeros(4)
    sin = np.ones(4)
    assert np.allclose(apply_rope_rotate_half(x, cos, sin), np.array([-3.0, -4.0, 1.0, 2.0]))


test_apply_rope_rotate_half()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def apply_rope_rotate_half(x, cos, sin):
    x = np.asarray(x, dtype=np.float64)
    half = x.shape[-1] // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * np.asarray(cos) + rotated * np.asarray(sin)
```

### 解析

1. 有些实现按偶奇维成对旋转，MiniMind 这里是 half split 形式。
2. cos/sin 来自预计算 position embedding。


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- 二维旋转：$(x_1,x_2)\mapsto(x_1\cos\theta-x_2\sin\theta,\ x_1\sin\theta+x_2\cos\theta)$。
- MiniMind 写法：$rope(x)=x\odot cos + rotate\_half(x)\odot sin$。
- RoPE 作用于 Q/K，使 attention score 依赖相对位置差。

### 易错点

- 偶奇配对和前后半维配对是两种实现约定，不能混用。
- cos/sin 的 seq 维、head 维广播错，会导致 batch 测试不稳定。
- KV cache 下 position 从 0 重启，会让新旧 token 的相对位置错乱。

### 面试追问

::: details 参考回答：RoPE 为什么能表达相对位置信息？

同一旋转矩阵族作用到 Q/K 后，两者点积会出现位置角度差项。也就是说 attention score 不只取决于内容向量，还取决于 query 和 key 的相对距离。

:::

::: details 参考回答：为什么 MiniMind 对 Q/K 旋转，而不是对 V 旋转？

attention 权重由 Q/K 点积决定，位置信息需要影响“看谁”。V 是被权重加权求和的内容载体，通常不需要承担位置匹配。

:::
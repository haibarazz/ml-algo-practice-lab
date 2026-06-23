# MoE Router Aux Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/02_model_architecture/moe_router_aux_loss/moe_router_aux_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `model/model_minimind.py:148-176`

## 手写实现约束

允许 NumPy；只实现 scores/topk_idx 到标量 aux loss。

## 原理最小说明

MoE 路由先算每个 token 到各专家的概率，再取 top-k。aux loss 用实际 load 和平均 routing score 的乘积，鼓励专家负载更均衡。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    """TODO guided implementation."""
    # TODO 1: 对 topk_idx 做 one-hot
    # TODO 2: 按 token 维求平均得到 load
    # TODO 3: scores 按 token 求平均
    # TODO 4: 计算 num_experts * coef * sum(load * mean_score)
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    """TODO blank implementation."""
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


def test_moe_router_aux_loss():
    scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    topk_idx = np.array([[0], [1]])
    loss = moe_router_aux_loss(scores, topk_idx, num_experts=3, coef=0.01)
    assert np.allclose(loss, 0.0135)


test_moe_router_aux_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    scores = np.asarray(scores, dtype=np.float64)
    topk_idx = np.asarray(topk_idx, dtype=np.int64)
    one_hot = np.eye(num_experts)[topk_idx]
    load = one_hot.mean(axis=0)
    mean_score = scores.mean(axis=0)
    return float((load * mean_score).sum() * num_experts * coef)
```

### 解析

1. MiniMind 默认 num_experts_per_tok=1，因此 load shape 通常是 `[1, E]`。
2. 如果 top-k>1，要确认归一化口径。


:::

## 工程要点 / 面试追问

见 `notes.md`。
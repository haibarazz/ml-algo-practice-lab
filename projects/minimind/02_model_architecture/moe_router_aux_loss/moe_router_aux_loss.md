# MoE 路由辅助损失：让专家负载更均衡

拆解 MiniMind `MOEFeedForward` 的 top-k 路由、专家聚合和 aux loss。

## 学习目标

- 理解 MoE 中 router、expert、top-k weight 的角色。
- 掌握 MiniMind 如何用 `index_add_` 聚合专家输出。
- 理解 aux loss 为什么要约束专家负载均衡。

## MiniMind 源码定位

- `model/model_minimind.py:148-176`

## 源码机制详解

`MOEFeedForward` 先把 `[batch, seq, hidden]` 展平为 `[tokens, hidden]`，router 对每个 token 输出 expert 概率。`topk` 选出每个 token 要走的专家，并可把 top-k 概率重新归一化。
每个 expert 只处理被分配给自己的 token。MiniMind 用 `token_idx` 找到这些 token，再把 `expert(x[token_idx]) * weight` 通过 `index_add_` 加回总输出。
训练时如果没有约束，router 可能长期偏向少数 expert，导致其他 expert 学不到东西。MiniMind 的 aux loss 用实际 load 和平均 router score 的乘积鼓励专家使用更均衡。

## 关键公式与数据流

- $p(e|x)=softmax(W_rx)$，选择 top-k expert。
- $y_i=\sum_{e\in topk(i)} w_{i,e} Expert_e(x_i)$。
- $L_{aux}=coef \cdot E \cdot \sum_e load_e \cdot score_e$。

## 为什么练这个

- 手写 aux loss 能理解 MoE 不只是多个 FFN，还包括路由训练问题。
- 这个练习对应真实源码中训练时额外加到 logits loss 上的 `aux_loss`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def moe_router_aux_loss(scores, topk_idx, num_experts, coef):
    """带提示实现。"""
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


def test_moe_router_aux_loss():
    scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    topk_idx = np.array([[0], [1]])
    loss = moe_router_aux_loss(scores, topk_idx, num_experts=3, coef=0.01)
    assert np.allclose(loss, 0.0135)


test_moe_router_aux_loss()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

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

## 工程要点 / 面试追问

### 关键公式与数据流

- $p(e|x)=softmax(W_rx)$，选择 top-k expert。
- $y_i=\sum_{e\in topk(i)} w_{i,e} Expert_e(x_i)$。
- $L_{aux}=coef \cdot E \cdot \sum_e load_e \cdot score_e$。

### 易错点

- top-k 概率不归一化时，多 expert 输出尺度可能漂移。
- 专家没有收到 token 时仍要保持 DDP/compile 图稳定。
- aux loss 只在训练时有意义，推理时不应影响输出。

### 面试追问

::: details 参考回答：MoE 为什么需要辅助负载均衡损失？

router 如果只选少数专家，会造成热门专家过载、冷门专家无梯度，模型容量没有真正用起来。辅助损失给 router 一个均衡使用专家的训练信号。

:::

::: details 参考回答：MiniMind 的 MoE 与普通 FFN 的输出维度有什么关系？

MoE 最终仍要输出 hidden size，才能接回 Transformer 残差。区别只是中间计算由一个 FFN 变成按 token 路由到多个 expert 后再聚合。

:::

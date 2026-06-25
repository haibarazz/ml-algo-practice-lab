# 逐 token logprob：rollout 后如何评估生成概率

复刻 `rollout_engine.compute_per_token_logps`，理解 RL/GRPO/PPO 中生成 token 的概率记录。

## 学习目标

- 理解为什么 rollout 后还要重新计算生成 token 的 logprob。
- 掌握 `logits_to_keep=n_keep+1` 与 shift 的关系。
- 理解 per-token logprob 如何服务 PPO/GRPO 的 ratio。

## MiniMind 源码定位

- `trainer/rollout_engine.py:23-36`
- `trainer/rollout_engine.py:71-92`

## 源码机制详解

`compute_per_token_logps` 接收完整 `input_ids`，只保留最后 `n_keep` 个生成 token 的 logprob。它 forward 时传 `logits_to_keep=n_keep+1`，因为要用前一个位置的 logits 预测后一个生成 token。
函数对 logits 做 log-softmax，再按真实 token id gather 出对应 logprob。返回形状是 `[batch, n_keep]`，每个元素表示模型在当时上下文下给这个生成 token 的 log 概率。
在 `TorchRolloutEngine.rollout` 中，模型先 generate 出 completion，再用当前 policy 重新算 per-token logprob。后续 PPO/GRPO 会比较 old/new/ref logprob，构造 ratio、KL 或 advantage 加权目标。

## 关键公式与数据流

- $\log p_\theta(y_t|x,y_{<t})=\log softmax(logits_{t-1})[y_t]$。
- $ratio_t=\exp(\log p_\theta(y_t)-\log p_{old}(y_t))$。

## 为什么练这个

- 这个练习是 PPO/GRPO 的前置模块：没有 per-token logprob 就没有 ratio。
- 它也训练你处理 logits 和 token id 的 gather 维度。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def per_token_logps(logits, token_ids):
    """带提示实现。"""
    # TODO 1: 对 vocab 维做稳定 log_softmax
    # TODO 2: 按 token_ids gather
    # TODO 3: 返回每个位置的 logprob
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def per_token_logps(logits, token_ids):
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


def test_per_token_logps():
    logits = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    ids = np.array([[0, 1]])
    out = per_token_logps(logits, ids)
    expected = np.array([[-np.log1p(np.exp(-2.0)), -np.log1p(np.exp(-2.0))]])
    assert np.allclose(out, expected)


test_per_token_logps()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def per_token_logps(logits, token_ids):
    logits = np.asarray(logits, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    batch = np.arange(token_ids.shape[0])[:, None]
    pos = np.arange(token_ids.shape[1])[None, :]
    return log_probs[batch, pos, token_ids]
```

### 解析

1. logits 和 token_ids 的 seq_len 必须一致。
2. 真实代码会配合 logits_to_keep 减少计算。

## 工程要点 / 面试追问

### 关键公式与数据流

- $\log p_\theta(y_t|x,y_{<t})=\log softmax(logits_{t-1})[y_t]$。
- $ratio_t=\exp(\log p_\theta(y_t)-\log p_{old}(y_t))$。

### 易错点

- `n_keep` 少加 1 会拿不到预测第一个生成 token 的 logits。
- gather 维度必须是 vocab 维。
- 生成序列里的 pad token 需要后续 completion_mask 屏蔽。

### 面试追问

::: details 参考回答：为什么 rollout 后不直接使用 generate 时的概率？

有些推理后端不返回完整概率，或者需要用当前训练图重新计算可对齐的 logprob。重新 forward 可以确保 logprob 与当前 policy、mask 和 dtype 处理一致。

:::

::: details 参考回答：per-token logprob 和序列 logprob 有什么关系？

序列 logprob 通常是有效 token logprob 的和。PPO/GRPO 常保留逐 token 形式，因为 ratio、clip 和 mask 都可能按 token 粒度计算。

:::

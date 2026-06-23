# Per Token Log Probs

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/05_preference_alignment/per_token_logps/per_token_logps.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `trainer/rollout_engine.py:23-36`

## 手写实现约束

允许 NumPy。

## 原理最小说明

RL 阶段需要知道生成 token 在当前策略下的 logprob。MiniMind 只保留最后 n_keep 个 token，并从 log_softmax 后的 vocab 分布里 gather 对应 id。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def per_token_logps(logits, token_ids):
    """TODO guided implementation."""
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


def test_per_token_logps():
    logits = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    ids = np.array([[0, 1]])
    out = per_token_logps(logits, ids)
    expected = np.array([[-np.log1p(np.exp(-2.0)), -np.log1p(np.exp(-2.0))]])
    assert np.allclose(out, expected)


test_per_token_logps()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

见 `notes.md`。
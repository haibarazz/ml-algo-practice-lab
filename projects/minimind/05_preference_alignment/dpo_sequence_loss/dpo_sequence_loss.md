# DPO Sequence Loss

> Status: complete

## Source Mapping

- `trainer/train_dpo.py:25-50`
- `trainer/train_dpo.py:53-84`

## 手写实现约束

允许 NumPy；输入已经是 per-token logprob。

## 原理最小说明

MiniMind 把 chosen 和 rejected 拼成一个 batch。先用 mask 对 token logprob 求和，再把前半视为 chosen、后半视为 rejected，计算 policy 相对 reference 的 log-ratio 改善。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
    """TODO guided implementation."""
    # TODO 1: mask 后按序列求和
    # TODO 2: batch 前半/后半切成 chosen/rejected
    # TODO 3: 计算 pi_logratios 和 ref_logratios
    # TODO 4: 返回 -log sigmoid(beta * diff) 的均值
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
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


def test_dpo_sequence_loss():
    ref = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    pol_good = np.array([[-0.1, -0.1], [-2.0, -2.0]])
    pol_bad = np.array([[-2.0, -2.0], [-0.1, -0.1]])
    mask = np.ones_like(ref)
    assert dpo_sequence_loss(ref, pol_good, mask, beta=1.0) < dpo_sequence_loss(ref, pol_bad, mask, beta=1.0)


test_dpo_sequence_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (np.asarray(ref_log_probs) * np.asarray(mask)).sum(axis=1)
    policy_log_probs = (np.asarray(policy_log_probs) * np.asarray(mask)).sum(axis=1)
    half = len(ref_log_probs) // 2
    chosen_ref, reject_ref = ref_log_probs[:half], ref_log_probs[half:]
    chosen_pol, reject_pol = policy_log_probs[:half], policy_log_probs[half:]
    logits = (chosen_pol - reject_pol) - (chosen_ref - reject_ref)
    return float(np.mean(np.logaddexp(0.0, -beta * logits)))
```

### 解析

1. batch 顺序必须是 chosen 在前 rejected 在后。
2. mask 应该只覆盖 assistant 回复 token。

## 工程要点 / 面试追问

见 `notes.md`。

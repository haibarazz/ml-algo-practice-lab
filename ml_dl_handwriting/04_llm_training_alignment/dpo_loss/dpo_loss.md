# DPO Loss

> Status: complete

## 题源线索

- Topic: DPO loss。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

DPO 使用 policy 和 reference 在 chosen/rejected 上的 log-prob 差：

$$L=-\log\sigma(\beta[(\log\pi_c-\log\pi_r)-(\log\pi^0_c-\log\pi^0_r)])$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
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
def test_dpo_loss():
    loss_good = dpo_loss([3.0], [1.0], [2.0], [1.5], beta=1.0)
    loss_bad = dpo_loss([1.0], [3.0], [2.0], [1.5], beta=1.0)
    assert loss_good < loss_bad
    assert np.isfinite(dpo_loss([1000.0], [0.0], [0.0], [0.0]))

test_dpo_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def _logsigmoid(x):
    return -np.logaddexp(0.0, -x)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    policy_chosen = np.asarray(policy_chosen, dtype=np.float64)
    policy_rejected = np.asarray(policy_rejected, dtype=np.float64)
    ref_chosen = np.asarray(ref_chosen, dtype=np.float64)
    ref_rejected = np.asarray(ref_rejected, dtype=np.float64)
    logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return np.mean(-_logsigmoid(logits))
```

### 解析

1. 先算 policy preference margin。
2. 再减 reference preference margin。
3. 乘 beta 控制偏好强度。
4. 用稳定 logsigmoid。

## 工程要点 / 面试追问

见 `notes.md`。

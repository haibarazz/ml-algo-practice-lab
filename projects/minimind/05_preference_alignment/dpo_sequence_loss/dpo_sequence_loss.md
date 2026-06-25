# DPO 序列级损失：让 chosen 相对 rejected 更可能

拆解 MiniMind `train_dpo.py` 中 reference/policy logprob margin 的计算。

## 学习目标

- 掌握 DPO loss 的序列级 logprob 求和。
- 理解 policy margin 和 reference margin 的差值。
- 理解 beta 如何控制偏好更新强度。

## MiniMind 源码定位

- `trainer/train_dpo.py:25-50`
- `trainer/train_dpo.py:53-85`

## 源码机制详解

`logits_to_log_probs` 先对 vocab 维做 log-softmax，再用 labels gather 出每个真实 token 的 logprob。`dpo_loss` 把 token logprob 乘 mask 后按序列求和，得到每条回答的条件 logprob。
batch 前半是 chosen，后半是 rejected。MiniMind 先算 policy 下的 `chosen - rejected`，再算 reference 下的 `chosen - rejected`，两者相减得到 DPO logits。
最终 loss 是 `-logsigmoid(beta * logits)`。当 policy 相比 reference 更偏向 chosen 时，logits 变大，loss 变小；反之则给模型梯度，推动 chosen 的相对概率上升。

## 关键公式与数据流

- $\Delta_\pi=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$。
- $\Delta_{ref}=\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x)$。
- $L=-\log\sigma(\beta(\Delta_\pi-\Delta_{ref}))$。

## 为什么练这个

- 手写 DPO loss 可以把偏好优化从公式落到 batch 切片。
- 这个模块也是理解 RLHF 替代路线的关键入口。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def dpo_sequence_loss(ref_log_probs, policy_log_probs, mask, beta):
    """带提示实现。"""
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


def test_dpo_sequence_loss():
    ref = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    pol_good = np.array([[-0.1, -0.1], [-2.0, -2.0]])
    pol_bad = np.array([[-2.0, -2.0], [-0.1, -0.1]])
    mask = np.ones_like(ref)
    assert dpo_sequence_loss(ref, pol_good, mask, beta=1.0) < dpo_sequence_loss(ref, pol_bad, mask, beta=1.0)


test_dpo_sequence_loss()
print("All tests passed.")
```

## 先停在这里

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

### 关键公式与数据流

- $\Delta_\pi=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$。
- $\Delta_{ref}=\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x)$。
- $L=-\log\sigma(\beta(\Delta_\pi-\Delta_{ref}))$。

### 易错点

- chosen/rejected batch 顺序写反会把偏好信号反过来。
- 忘记 reference margin 会退化成简单偏好分类，约束变弱。
- logprob 应只统计回答 token，不能把 prompt 也算进去。

### 面试追问

::: details 参考回答：DPO 里的 reference model 起什么作用？

reference model 提供原模型的偏好基线，DPO 优化的是 policy 相对 reference 的偏好提升。它相当于隐式约束模型不要偏离原能力太远。

:::

::: details 参考回答：beta 变大或变小有什么影响？

beta 越大，同样的 margin 差产生更强梯度，训练更激进；beta 越小，更新更温和。过大可能过拟合偏好数据，过小可能学不动。

:::

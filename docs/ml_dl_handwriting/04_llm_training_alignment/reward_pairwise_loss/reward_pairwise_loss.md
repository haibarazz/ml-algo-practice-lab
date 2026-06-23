# Reward Pairwise Loss

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/04_llm_training_alignment/reward_pairwise_loss/reward_pairwise_loss.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Reward Model 常用 pairwise loss，让 chosen reward 大于 rejected reward：

$$L=-\log\sigma(r_c-r_r)$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def reward_pairwise_loss(chosen_rewards, rejected_rewards):
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

def reward_pairwise_loss(chosen_rewards, rejected_rewards):
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
def test_reward_pairwise_loss():
    assert reward_pairwise_loss([3.0], [1.0]) < reward_pairwise_loss([1.0], [3.0])
    loss = reward_pairwise_loss([1.0, 2.0], [0.0, 3.0])
    assert np.isfinite(loss)

test_reward_pairwise_loss()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def reward_pairwise_loss(chosen_rewards, rejected_rewards):
    chosen_rewards = np.asarray(chosen_rewards, dtype=np.float64)
    rejected_rewards = np.asarray(rejected_rewards, dtype=np.float64)
    diff = chosen_rewards - rejected_rewards
    return np.mean(np.logaddexp(0.0, -diff))
```

### 解析

1. 只关心 chosen 和 rejected 的差。
2. 差越大，loss 越小。
3. `logaddexp(0, -diff)` 是 `-logsigmoid(diff)` 的稳定写法。


:::

## 工程要点 / 面试追问

### 核心公式

- Bradley-Terry 形式：$P(y_w \succ y_l)=\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))$。
- pairwise loss：$L=-\log\sigma(r_w-r_l)$。

### 易错点

- 用 MSE 拟合绝对分数，而不是学习相对偏好。
- `chosen`/`rejected` 顺序反了。
- reward 输出尺度无约束，后续 RL 或排序时需要注意校准和 clipping。
- 只看 pair accuracy 不够，还要关注 reward hacking 和分布外泛化。

### 面试追问

- Reward model 为什么可以只学相对偏好？
- pairwise、pointwise、listwise ranking loss 有什么区别？
- reward 分数的绝对值有意义吗？
- reward model 如何被用于 PPO/RLHF？
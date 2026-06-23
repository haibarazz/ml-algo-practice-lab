# Entropy And KL

> Status: complete

## 题源线索

- Topic: 实现熵和 KL 散度。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

离散熵：

$$H(p)=-\sum_i p_i\log p_i$$

KL 散度：

$$D_{KL}(p||q)=\sum_i p_i\log\frac{p_i}{q_i}$$

实现时要避免 `log(0)`，通常使用 `eps` 做裁剪。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
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

def entropy_kl(p, q=None, eps=1e-12):
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
def test_entropy_kl():
    assert np.allclose(entropy_kl([0.5, 0.5]), np.log(2))
    entropy, kl = entropy_kl([0.5, 0.5], [0.25, 0.75])
    assert np.allclose(entropy, np.log(2))
    assert kl > 0
    assert np.allclose(entropy_kl([1.0, 0.0]), 0.0, atol=1e-8)

test_entropy_kl()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def entropy_kl(p, q=None, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = p / np.sum(p)
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe))
    if q is None:
        return entropy
    q = np.asarray(q, dtype=np.float64)
    q = q / np.sum(q)
    q_safe = np.clip(q, eps, 1.0)
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))
    return entropy, kl
```

### 解析

1. 输入先归一化，避免调用方给的是未归一化权重。
2. `eps` 裁剪用于避免 `log(0)`。
3. KL 不对称，`KL(p||q)` 和 `KL(q||p)` 通常不同。

## 工程要点 / 面试追问

见 `notes.md`。

# Dropout Train Eval

> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

Inverted Dropout 在训练时随机置零并除以 keep probability，推理时直接返回输入。

$$y = x \cdot mask / (1-p)$$

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def dropout_train_eval(X, p=0.5, training=True, seed=None):
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

def dropout_train_eval(X, p=0.5, training=True, seed=None):
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
def test_dropout_train_eval():
    X = np.ones(1000)
    out = dropout_train_eval(X, p=0.2, training=True, seed=0)
    assert set(np.unique(out)).issubset({0.0, 1.25})
    assert abs(out.mean() - 1.0) < 0.1
    assert np.allclose(dropout_train_eval(X, p=0.2, training=False), X)

test_dropout_train_eval()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def dropout_train_eval(X, p=0.5, training=True, seed=None):
    X = np.asarray(X, dtype=np.float64)
    if not training:
        return X
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 - p
    mask = (rng.random(X.shape) < keep_prob).astype(np.float64)
    return X * mask / keep_prob
```

### 解析

1. 训练时用随机 mask。
2. 除以 keep_prob 保持期望不变。
3. 推理时不做随机丢弃。

## 工程要点 / 面试追问

### 核心公式

- inverted dropout 训练态：$y=\frac{m\odot x}{1-p}$，$m\sim Bernoulli(1-p)$。
- 推理态通常直接 $y=x$。

### 易错点

- 推理时还随机丢弃，导致输出不稳定。
- 训练时没有除以 keep probability，导致期望变小。
- `p` 和 `keep_prob` 混淆。
- mask 需要和输入 shape 可广播，并且通常只在训练态采样。

### 面试追问

- Dropout 为什么能缓解过拟合？
- 什么是 inverted dropout？为什么推理时不用再缩放？
- Dropout 和 BatchNorm 同时使用有什么注意点？
- Transformer 中 attention dropout 和 FFN dropout 分别作用在哪里？

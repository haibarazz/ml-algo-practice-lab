# Causal LM Shift Loss：用当前位置预测下一个 token

拆解 MiniMindForCausalLM 里 logits 与 labels 的错位交叉熵。

## 学习目标

- 理解 next-token prediction 的 logits/labels shift。
- 掌握 flatten 后计算 cross entropy 的形状变化。
- 理解 ignore_index 怎样同时服务 pad 和 SFT mask。

## MiniMind 源码定位

- `model/model_minimind.py:245-253`
- `trainer/train_pretrain.py:24-80`

## 源码机制详解

`MiniMindForCausalLM.forward` 先得到 hidden states，再通过 `lm_head` 投影到 vocab logits。如果传入 labels，它会令 `x = logits[..., :-1, :]`，`y = labels[..., 1:]`，也就是第 t 个位置预测第 t+1 个 token。
随后把 `[batch, seq-1, vocab]` 展平成 `[batch*(seq-1), vocab]`，把标签展平成 `[batch*(seq-1)]`，交给 `F.cross_entropy`。`ignore_index=-100` 会跳过 pad 或非 assistant 位置。
预训练、SFT 和部分对齐训练都共用这个 causal LM 基础。区别不在模型 loss 公式，而在 dataset 给 labels 哪些位置填真实 token、哪些位置填 `-100`。

## 关键公式与数据流

- $L=-\frac{1}{N}\sum_{t\in valid}\log p_\theta(x_{t+1}|x_{\le t})$。
- $x=logits_{0:T-1}$，$y=labels_{1:T}$。
- 若 $labels_t=-100$，该位置不计入 $N$ 和 loss。

## 为什么练这个

- 手写 shift loss 能把 LLM 最基础训练目标讲清楚。
- 这个模块连接数据集里的 labels 和模型内部的 loss。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    """带提示实现。"""
    # TODO 1: shift logits 和 labels
    # TODO 2: flatten batch/seq
    # TODO 3: 过滤 ignore_index
    # TODO 4: 稳定 softmax 交叉熵
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def causal_lm_shift_loss(logits, labels, ignore_index=-100):
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


def test_causal_lm_shift_loss():
    logits = np.array([[[0, 5, 0], [0, 0, 5], [5, 0, 0]]], dtype=float)
    labels = np.array([[0, 1, 2]])
    assert causal_lm_shift_loss(logits, labels) < 0.02
    masked = np.array([[0, -100, 2]])
    assert np.isfinite(causal_lm_shift_loss(logits, masked))


test_causal_lm_shift_loss()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def causal_lm_shift_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    x = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    y = labels[:, 1:].reshape(-1)
    mask = y != ignore_index
    x = x[mask]
    y = y[mask]
    shifted = x - np.max(x, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    correct = shifted[np.arange(len(y)), y]
    return float(np.mean(-correct + logsumexp))
```

### 解析

1. logits 和 labels 的错位是最常见 bug。
2. ignore_index 是为了屏蔽 pad 或非 assistant token。

## 工程要点 / 面试追问

### 关键公式与数据流

- $L=-\frac{1}{N}\sum_{t\in valid}\log p_\theta(x_{t+1}|x_{\le t})$。
- $x=logits_{0:T-1}$，$y=labels_{1:T}$。
- 若 $labels_t=-100$，该位置不计入 $N$ 和 loss。

### 易错点

- 不 shift 会变成预测当前 token，模型能作弊。
- shift 两次会让监督错位。
- flatten 前后 batch/seq 顺序必须一致。

### 面试追问

::: details 参考回答：为什么 causal LM 要 shift 一位？

位置 t 的 hidden state 只能看见 t 及之前的 token，所以它的监督目标应该是下一个 token。shift 一位正是把输入上下文和下一个 token 标签对齐。

:::

::: details 参考回答：`ignore_index=-100` 在预训练和 SFT 中分别屏蔽什么？

预训练中主要屏蔽 padding；SFT 中还屏蔽 system/user/prompt 等非 assistant 位置。两者都通过同一个 cross entropy 参数生效。

:::

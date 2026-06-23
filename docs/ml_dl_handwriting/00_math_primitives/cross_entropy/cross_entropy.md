# Cross Entropy

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/ml_dl_handwriting/00_math_primitives/cross_entropy/cross_entropy.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## 手写实现约束

允许使用 Python 基础语法和 NumPy；不允许调用 sklearn、torch 或现成算法实现。

## 原理最小说明

多分类交叉熵常和 softmax 一起使用。对第 `i` 个样本：

$$L_i=-\log\frac{e^{z_{i,y_i}}}{\sum_c e^{z_{i,c}}}$$

为了数值稳定，先减去每行最大 logit，再计算 log-sum-exp。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def cross_entropy(logits, targets):
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

def cross_entropy(logits, targets):
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
def test_cross_entropy():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    targets = np.array([0, 1])
    loss = cross_entropy(logits, targets)
    assert np.allclose(loss, 0.288725992, atol=1e-6)
    large = np.array([[1000.0, 1001.0]])
    assert np.isfinite(cross_entropy(large, np.array([1])))

test_cross_entropy()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
import numpy as np

def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct = shifted[np.arange(logits.shape[0]), targets]
    return np.mean(-correct + logsumexp)
```

### 解析

1. 减最大值避免 `exp(1000)` 溢出。
2. `correct` 取出目标类别对应的 shifted logit。
3. 每个样本损失是 `-correct + logsumexp`。
4. 最后对 batch 求平均。


:::

## 工程要点 / 面试追问

### 核心公式

- one-hot target 下：$L=-\sum_c y_c\log p_c$；若 target 是类别下标 $t$，则 $L=-\log p_t$。
- 对 logits 使用 softmax + CE 时，常见梯度为 $\frac{\partial L}{\partial z}=p-y$。

### 易错点

- 先 softmax 再 log 容易数值不稳定，应使用 log-sum-exp 或 fused cross entropy。
- target 下标和 one-hot 表示混用，导致 shape 或语义错误。
- 忘记 batch mean，使 loss 尺度随 batch size 变化。
- 把多分类单标签 CE 和多标签 BCE 混淆。

### 面试追问

::: details 参考回答：softmax + cross entropy 的梯度为什么可以化简成 `p - y`？

对 logits 求导时，softmax 的雅可比矩阵和 `-log p_y` 的梯度会合并，最终得到 `p - y`。这个结果直观上表示预测概率比目标概率高的类别要被压低，目标类别要被抬高。

:::

::: details 参考回答：为什么 PyTorch 的 `CrossEntropyLoss` 输入通常是 logits 而不是概率？

logits 可以进入 fused cross entropy，用 log-sum-exp 做稳定计算，避免先 softmax 后 log 的数值问题。如果输入已经是概率，极小概率会导致 `log(0)` 或梯度不稳定，而且还可能重复 softmax。

:::

::: details 参考回答：label smoothing 会怎样改变 target 分布和梯度？

label smoothing 会把 one-hot 目标改成“目标类接近 1、非目标类有少量概率”的软标签。这样梯度不再把正确类概率强推到 1，可以减轻过度自信，常对泛化和校准有帮助。

:::

::: details 参考回答：类别不均衡时，cross entropy 可以怎样加权？

可以给不同类别设置 class weight，让少数类样本的 loss 权重更大。工程上也常结合重采样、focal loss 或按业务成本设置权重，但要注意验证集指标是否真的改善。

:::
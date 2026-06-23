# MLP Backward

> Status: sample complete

## 手写实现约束

允许使用 NumPy；不允许使用 torch autograd、torch.nn、keras 或其他自动微分框架。

## 原理最小说明

本模块实现一个两层 MLP：

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = ReLU(Z_1)
$$

$$
Y = A_1W_2 + b_2
$$

损失使用 MSE：

$$
L = \frac{1}{N}\sum (Y - T)^2
$$

反向传播的核心是链式法则：从 `dY` 开始，依次求 `dW2/db2/dA1/dZ1/dW1/db1`。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np


def mlp_forward_backward_guided(X, y, W1, b1, W2, b2):
    """Forward and backward for a two-layer MLP with ReLU and MSE."""
    # TODO 1: forward
    # z1 = ?
    # a1 = ?
    # pred = ?
    # loss = ?

    # TODO 2: start backward from MSE loss
    # dpred = ?

    # TODO 3: gradients for second linear layer
    # dW2 = ?
    # db2 = ?
    # da1 = ?

    # TODO 4: backprop through ReLU
    # dz1 = ?

    # TODO 5: gradients for first linear layer
    # dW1 = ?
    # db1 = ?

    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np


def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Forward and backward for a two-layer MLP with ReLU and MSE.

    Returns:
        loss: scalar float
        grads: dict with dW1, db1, dW2, db2
    """
    raise NotImplementedError
```

## 测试区

运行：

```bash
python tests.py
```

Notebook 中可以在实现 `blank` 函数后直接运行：

```python
def test_mlp_forward_backward():
    X = np.array([[1.0, -1.0], [0.5, 2.0]])
    y = np.array([[1.0], [0.0]])
    W1 = np.array([[0.2, -0.4, 0.1], [0.7, 0.3, -0.5]])
    b1 = np.array([0.0, 0.1, -0.2])
    W2 = np.array([[0.6], [-0.1], [0.2]])
    b2 = np.array([0.05])

    loss, grads = mlp_forward_backward(X, y, W1, b1, W2, b2)
    assert np.isscalar(loss) or np.asarray(loss).shape == ()
    assert set(grads) == {"dW1", "db1", "dW2", "db2"}
    assert grads["dW1"].shape == W1.shape
    assert grads["db1"].shape == b1.shape
    assert grads["dW2"].shape == W2.shape
    assert grads["db2"].shape == b2.shape

    expected_loss = 0.78345
    assert np.allclose(loss, expected_loss, atol=1e-5)
    assert np.allclose(grads["dW2"], np.array([[1.35], [0.45], [-0.348]]), atol=1e-5)
    assert np.allclose(grads["db2"], np.array([0.03]), atol=1e-5)
    assert np.allclose(grads["dW1"], np.array([[0.27, -0.045, -0.174], [1.08, -0.18, 0.174]]), atol=1e-5)
    assert np.allclose(grads["db1"], np.array([0.54, -0.09, -0.174]), atol=1e-5)


test_mlp_forward_backward()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np


def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Forward and backward for a two-layer MLP with ReLU and MSE.

    Returns:
        loss: scalar float
        grads: dict with dW1, db1, dW2, db2
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    W1 = np.asarray(W1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    W2 = np.asarray(W2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    pred = a1 @ W2 + b2

    diff = pred - y
    loss = np.mean(diff * diff)

    dpred = 2.0 * diff / diff.size
    dW2 = a1.T @ dpred
    db2 = dpred.sum(axis=0)
    da1 = dpred @ W2.T

    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return loss, grads
```

### 解析

1. MSE 的平均范围是所有输出元素，因此 `dpred = 2 * diff / diff.size`。
2. 线性层 `Y = A @ W + b` 的梯度是 `dW = A.T @ dY`，`db = sum(dY)`。
3. ReLU backward 用 mask：`z1 > 0` 的位置梯度通过，否则为 0。
4. 所有梯度 shape 必须和参数 shape 完全一致，这是反向传播最基本的检查。

## 工程要点 / 面试追问

### 核心公式

- 链式法则：$\frac{\partial L}{\partial W_2}=h^\top\frac{\partial L}{\partial \hat y}$，$\frac{\partial L}{\partial h}=\frac{\partial L}{\partial \hat y}W_2^\top$。
- ReLU backward：$\frac{\partial h}{\partial z_1}=\mathbf{1}[z_1>0]$。

### 易错点

- MSE 梯度分母用 batch size 还是元素总数，要和 loss 定义一致。
- ReLU backward 推荐缓存 `z1`，不要依赖被后续修改过的激活值。
- `db` 忘记沿 batch 维求和。
- 矩阵乘法方向写反，导致 shape 对不上或梯度转置。

### 面试追问

::: details 参考回答：如果输出层改成 softmax + cross entropy，最后一层梯度如何化简？

softmax + cross entropy 时，最后一层 logits 的梯度可以直接写成 `probs - onehot`，再按 batch 或 token 平均。这样不用显式写 softmax 的完整雅可比矩阵，数值和实现都更稳定。

:::

::: details 参考回答：为什么需要激活函数？没有激活时 backward 会发生什么？

没有激活函数时，整个网络只是线性层的复合，forward 等价于一个线性模型。backward 仍然能算，但训练再深也不会增加非线性表达能力，只会让参数冗余和优化更复杂。

:::

::: details 参考回答：梯度消失和梯度爆炸分别可能出现在哪里？

梯度消失常出现在深层链式乘法中，尤其是 sigmoid/tanh 饱和区或不合适初始化下。梯度爆炸也来自多层雅可比连乘，RNN、深层网络或学习率过大时更明显。

:::

::: details 参考回答：如何用数值梯度检查 backward 是否正确？

数值梯度检查用有限差分近似某个参数的导数，再和解析梯度比较相对误差。检查时要用小网络、固定随机种子、关闭 dropout，并选择合适 epsilon，避免浮点误差主导。

:::

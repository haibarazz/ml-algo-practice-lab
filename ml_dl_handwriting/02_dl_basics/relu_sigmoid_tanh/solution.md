# Relu Sigmoid Tanh Solution

## Reference Implementation

```python
import numpy as np

def relu_sigmoid_tanh(x):
    x = np.asarray(x, dtype=np.float64)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    tanh = np.tanh(x)
    relu = np.maximum(x, 0.0)
    return {
        "relu": relu,
        "relu_grad": (x > 0).astype(np.float64),
        "sigmoid": sigmoid,
        "sigmoid_grad": sigmoid * (1 - sigmoid),
        "tanh": tanh,
        "tanh_grad": 1 - tanh * tanh,
    }
```

## Explanation

1. ReLU 梯度在 0 处不可导，本模块约定为 0。
2. sigmoid 梯度可写成 `s * (1-s)`。
3. tanh 梯度是 `1 - tanh^2`。

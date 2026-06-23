# FFN Relu Gelu Solution

## Reference Implementation

```python
import numpy as np

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def ffn_relu_gelu(X, W1, b1, W2, b2, activation='relu'):
    hidden = X @ W1 + b1
    if activation == 'relu':
        hidden = np.maximum(hidden, 0.0)
    elif activation == 'gelu':
        hidden = _gelu(hidden)
    else:
        raise ValueError("activation must be relu or gelu")
    return hidden @ W2 + b2
```

## Explanation

1. FFN 对每个 token 独立作用。
2. ReLU 直接截断负值。
3. GeLU 常用 tanh 近似。
4. 最后一层投回模型维度。

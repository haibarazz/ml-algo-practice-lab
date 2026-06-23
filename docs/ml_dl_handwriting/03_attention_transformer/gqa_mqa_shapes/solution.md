# GQA MQA Shapes Solution

## Reference Implementation

```python
import numpy as np

def _repeat_kv(x, target_heads):
    b, kv_heads, seq, dim = x.shape
    assert target_heads % kv_heads == 0
    repeat = target_heads // kv_heads
    return np.repeat(x, repeat, axis=1)

def gqa_mqa_shapes(q, k, v):
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    target_heads = q.shape[1]
    return _repeat_kv(k, target_heads), _repeat_kv(v, target_heads)
```

## Explanation

1. 输入约定 `[batch, heads, seq, head_dim]`。
2. query heads 是目标 head 数。
3. K/V heads 通过 repeat 复用。
4. target_heads 必须能整除 kv_heads。

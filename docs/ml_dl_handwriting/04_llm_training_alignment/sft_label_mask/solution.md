# SFT Label Mask Solution

## Reference Implementation

```python
import numpy as np

def sft_label_mask(input_ids, prompt_lengths, pad_token_id=0, ignore_index=-100):
    input_ids = np.asarray(input_ids, dtype=np.int64)
    labels = input_ids.copy()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = ignore_index
    labels[input_ids == pad_token_id] = ignore_index
    return labels
```

## Explanation

1. labels 初始复制 input_ids。
2. 每条样本按 prompt length 屏蔽 prompt。
3. padding 统一屏蔽。
4. 训练时 CE 会忽略 -100。

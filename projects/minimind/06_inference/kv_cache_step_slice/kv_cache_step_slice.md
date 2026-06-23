# KV Cache Step Slice

> Status: complete

## Source Mapping

- `model/model_minimind.py:257-288`
- `model/model_minimind.py:120-124`
- `model/model_minimind.py:209-232`

## 手写实现约束

只用 Python list。

## 原理最小说明

启用 KV cache 后，历史 token 已经缓存了 K/V。下一步 forward 只需要新 token，因此 MiniMind 用 `input_ids[:, past_len:]` 作为本步输入。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def slice_new_tokens(input_ids, past_len):
    """TODO guided implementation."""
    # TODO 1: 输入是二维 list
    # TODO 2: 对每行保留 past_len 之后的 token
    # TODO 3: past_len 为 0 时返回完整输入
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def slice_new_tokens(input_ids, past_len):
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
def test_slice_new_tokens():
    ids = [[1, 5, 6], [1, 7, 8]]
    assert slice_new_tokens(ids, 0) == ids
    assert slice_new_tokens(ids, 2) == [[6], [8]]


test_slice_new_tokens()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def slice_new_tokens(input_ids, past_len):
    return [list(row[past_len:]) for row in input_ids]
```

### 解析

1. KV cache 不是缓存 logits，而是每层 attention 的 K/V。
2. attention_mask 仍要随生成长度增长。

## 工程要点 / 面试追问

见 `notes.md`。

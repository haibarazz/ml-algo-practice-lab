# DPO Pair Shift

> Status: complete

## Source Mapping

- `dataset/lm_dataset.py:122-192`

## 手写实现约束

只用 Python list。

## 原理最小说明

DPO 数据一条样本有 chosen 和 rejected。训练时仍是 next-token 形式，所以每条序列都要切成 `x=input[:-1]`、`y=input[1:]`，mask 也对齐到 y 位置。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    """TODO guided implementation."""
    # TODO 1: chosen/rejected 分别切 x 和 y
    # TODO 2: mask 去掉第一个位置
    # TODO 3: 返回统一字典
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
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
def test_build_dpo_shifted():
    out = build_dpo_shifted([1, 4, 5, 2], [1, 4, 6, 2], [0, 1, 1, 1], [0, 1, 1, 1])
    assert out["x_chosen"] == [1, 4, 5]
    assert out["y_chosen"] == [4, 5, 2]
    assert out["mask_chosen"] == [1, 1, 1]
    assert out["x_rejected"] == [1, 4, 6]
    assert out["y_rejected"] == [4, 6, 2]


test_build_dpo_shifted()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    return {
        "x_chosen": list(chosen_ids[:-1]),
        "y_chosen": list(chosen_ids[1:]),
        "mask_chosen": list(chosen_mask[1:]),
        "x_rejected": list(rejected_ids[:-1]),
        "y_rejected": list(rejected_ids[1:]),
        "mask_rejected": list(rejected_mask[1:]),
    }
```

### 解析

1. mask 和 y 对齐，不是和 x 对齐。
2. chosen/rejected 顺序会影响 DPO loss 的符号。

## 工程要点 / 面试追问

见 `notes.md`。

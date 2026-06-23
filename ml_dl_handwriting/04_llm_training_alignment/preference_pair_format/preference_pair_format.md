# Preference Pair Format

> Status: complete

## 题源线索

- Topic: chosen / rejected 偏好数据组织。
- Source index: `source-research/niuke-ml-dl-topic-index.md`

## 手写实现约束

允许使用 list / dict；不调用数据集框架。

## 原理最小说明

DPO/RM 常用偏好对：同一个 prompt 下有 chosen 和 rejected 两个回答。标准化数据结构能减少后续 collator 和 loss 的混乱。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python


def preference_pair_format(prompts, chosen, rejected):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python


def preference_pair_format(prompts, chosen, rejected):
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
def test_preference_pair_format():
    pairs = preference_pair_format(["Q1", "Q2"], ["good1", "good2"], ["bad1", "bad2"])
    assert pairs[0] == {"prompt": "Q1", "chosen": "good1", "rejected": "bad1"}
    try:
        preference_pair_format(["Q"], ["A"], [])
        assert False
    except ValueError:
        pass

test_preference_pair_format()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
def preference_pair_format(prompts, chosen, rejected):
    if not (len(prompts) == len(chosen) == len(rejected)):
        raise ValueError("prompts, chosen, rejected must have the same length")
    pairs = []
    for p, c, r in zip(prompts, chosen, rejected):
        pairs.append({"prompt": p, "chosen": c, "rejected": r})
    return pairs
```

### 解析

1. 三个列表长度必须一致。
2. 每条样本保留 prompt/chosen/rejected 三个字段。
3. 这是后续 DPO/RM 的最小数据单元。

## 工程要点 / 面试追问

见 `notes.md`。

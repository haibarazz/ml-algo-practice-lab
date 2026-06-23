# Empty Think Cleanup

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/04_sft/empty_think_cleanup/empty_think_cleanup.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `dataset/lm_dataset.py:31-35`
- `dataset/lm_dataset.py:106-119`
- `dataset/lm_dataset.py:195-224`

## 手写实现约束

只用 Python string；把随机行为改成显式参数。

## 原理最小说明

MiniMind 数据中可能有空 `<think>` 块。训练时会按概率移除空思考标签，避免模型过度学习空 thinking 模板。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def cleanup_empty_think(prompt, keep_empty=False):
    """TODO guided implementation."""
    # TODO 1: 定义空 think 模式
    # TODO 2: keep_empty 为 True 时原样返回
    # TODO 3: 否则移除空 think 模式
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def cleanup_empty_think(prompt, keep_empty=False):
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
def test_cleanup_empty_think():
    text = "A<think>\n\n</think>\n\nB"
    assert cleanup_empty_think(text, keep_empty=True) == text
    assert cleanup_empty_think(text, keep_empty=False) == "AB"


test_cleanup_empty_think()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def cleanup_empty_think(prompt, keep_empty=False):
    pattern = "<think>\n\n</think>\n\n"
    if keep_empty:
        return prompt
    return prompt.replace(pattern, "")
```

### 解析

1. 真实代码里 keep/remove 是随机比例控制。
2. 这是数据增强/模板鲁棒性的一部分。


:::

## 工程要点 / 面试追问

见 `notes.md`。
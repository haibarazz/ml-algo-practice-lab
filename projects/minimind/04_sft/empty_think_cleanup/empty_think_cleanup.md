# 空 thinking 标签清理：控制推理风格和数据分布

拆解 `post_processing_chat` 和 RLAIF/SFT 数据中 `<think>` 空块的随机保留逻辑。

## 学习目标

- 理解 chat template 中 `<think>` 标签的训练语义。
- 掌握为什么空 thinking 块不能机械全部保留或全部删除。
- 理解数据增强式随机清理对输出格式的影响。

## MiniMind 源码定位

- `dataset/lm_dataset.py:31-35`
- `dataset/lm_dataset.py:195-224`
- `trainer/train_full_sft.py:24-80`

## 源码机制详解

MiniMind 的 chat template 可能生成 `<think>\n\n</think>\n\n` 这种空思考块。`post_processing_chat` 以一定概率移除它，让训练数据中同时存在显式空 thinking 和直接回答两种形式。
RLAIFDataset 还会通过 `open_thinking` 控制生成 prompt 是否打开 thinking。也就是说 MiniMind 不只是学回答内容，也在学“什么时候带 thinking 格式、什么时候不带”。
这个模块看起来像字符串替换，但它影响的是模型输出风格和角色模板分布。对开源 LLM 项目来说，模板清理常常比模型结构更容易造成可见行为差异。

## 关键公式与数据流

- 若随机数 $u>empty\_think\_ratio$，将空块 `<think>\\n\\n</think>\\n\\n` 替换为空串。
- 保留概率约为 $empty\_think\_ratio$，删除概率约为 $1-empty\_think\_ratio$。

## 为什么练这个

- 手写这个函数训练你关注 tokenizer 模板中的特殊标签。
- 它也提示 SFT 不是只有 loss，数据格式清理本身就是训练策略。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def cleanup_empty_think(prompt, keep_empty=False):
    """带提示实现。"""
    # TODO 1: 定义空 think 模式
    # TODO 2: keep_empty 为 True 时原样返回
    # TODO 3: 否则移除空 think 模式
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def cleanup_empty_think(prompt, keep_empty=False):
    """无提示实现。"""
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

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

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

## 工程要点 / 面试追问

### 关键公式与数据流

- 若随机数 $u>empty\_think\_ratio$，将空块 `<think>\\n\\n</think>\\n\\n` 替换为空串。
- 保留概率约为 $empty\_think\_ratio$，删除概率约为 $1-empty\_think\_ratio$。

### 易错点

- 机械删除所有 think 标签会改变模板分布。
- 只做字符串替换时要保证不会误删非空推理内容。
- 训练和推理模板不一致，会造成模型输出格式漂移。

### 面试追问

::: details 参考回答：为什么空 `<think>` 标签也值得单独处理？

它虽然没有语义内容，但会成为模型学习的输出格式。大量空 thinking 块可能让模型习惯输出空思考段，全部删除又可能失去对模板格式的适应。

:::

::: details 参考回答：这个函数和模型能力有什么关系？

它不改变模型结构，却改变训练分布。LLM 的行为很大一部分来自数据模板，格式清理会直接影响回答是否带 thinking、tool call 或角色标记。

:::

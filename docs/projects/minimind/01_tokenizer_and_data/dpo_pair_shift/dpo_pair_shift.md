# DPO 偏好样本：chosen/rejected 的 x、y、mask

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/01_tokenizer_and_data/dpo_pair_shift/dpo_pair_shift.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


复刻 `DPODataset` 怎样把一对偏好回答转换成 DPO 训练所需的 token 张量。

## 学习目标

- 理解 DPO 数据为什么必须成对包含 chosen 和 rejected。
- 掌握 `x=tokens[:-1]`、`y=tokens[1:]` 和 response mask 的对齐。
- 理解 DPO loss 只应该统计 assistant 回复 token。

## MiniMind 源码定位

- `dataset/lm_dataset.py:122-192`
- `trainer/train_dpo.py:53-85`

## 源码机制详解

`DPODataset` 读取同一样本中的 `chosen` 和 `rejected` 两段 conversation，分别套用同一个 chat template。这样两条序列共享 prompt 语义，差异集中在回答质量上。
它对每条序列都构造三件东西：`x` 是去掉最后一个 token 的输入，`y` 是去掉第一个 token 的下一个 token 标签，`mask` 标出 assistant 回复位置。注意 mask 也要从第二个位置开始对齐 `y`。
训练时 `train_dpo.py` 会把 chosen 和 rejected 在 batch 维拼接，先通过 reference model 和 policy model 分别得到每个 token 的 logprob，再按 mask 求和得到序列级 logprob。

## 关键公式与数据流

- $x = ids_{0:T-1}$，$y = ids_{1:T}$，$mask = response\_mask_{1:T}$。
- $\log p_\theta(y|x)=\sum_t mask_t \log p_\theta(y_t|x_{\le t})$。
- chosen/rejected 必须来自同一个 prompt，否则 logprob 差混入了问题难度差异。

## 为什么练这个

- 这个练习把 DPO 的数据前处理和后续序列级 loss 对齐起来。
- 手写切片和 mask 能避免 DPO 中最常见的 off-by-one bug。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    """带提示实现。"""
    # TODO 1: chosen/rejected 分别切 x 和 y
    # TODO 2: mask 去掉第一个位置
    # TODO 3: 返回统一字典
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_dpo_shifted(chosen_ids, rejected_ids, chosen_mask, rejected_mask):
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

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

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


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- $x = ids_{0:T-1}$，$y = ids_{1:T}$，$mask = response\_mask_{1:T}$。
- $\log p_\theta(y|x)=\sum_t mask_t \log p_\theta(y_t|x_{\le t})$。
- chosen/rejected 必须来自同一个 prompt，否则 logprob 差混入了问题难度差异。

### 易错点

- chosen/rejected 顺序反了，训练目标会鼓励坏回答。
- mask 没有跟着 shift，logprob 会统计错 token。
- chosen/rejected 使用不同模板，会让偏好比较不公平。

### 面试追问

::: details 参考回答：DPO 数据和 SFT 数据最核心的区别是什么？

SFT 是单个 prompt 对一个示范答案，目标是模仿；DPO 是同一 prompt 下 chosen/rejected 成对比较，目标是提高 chosen 相对 rejected 的概率优势。

:::

::: details 参考回答：为什么 DPO 里 chosen 和 rejected 要共享同一个 prompt？

只有共享 prompt，模型比较的才是回答质量差异。prompt 不同会把问题难度、长度和主题差异混进 logprob margin，偏好信号会变脏。

:::
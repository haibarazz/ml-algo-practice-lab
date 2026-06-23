# Binary AUC

> Status: complete

## 手写实现约束

允许使用 list / NumPy；不允许调用 sklearn.metrics.roc_auc_score。

## 原理最小说明

AUC 可以理解为：随机抽一个正样本和一个负样本，正样本得分高于负样本的概率。

成对实现：

- 正样本分数大于负样本：记 1
- 相等：记 0.5
- 小于：记 0

最后除以正负样本对数。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
import numpy as np

def auc_binary(y_true, y_score):
    """TODO guided implementation."""
    # TODO 1: prepare inputs and check shapes
    # TODO 2: implement the core formula
    # TODO 3: handle edge cases and return result
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
import numpy as np

def auc_binary(y_true, y_score):
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
def test_auc_binary():
    assert np.allclose(auc_binary([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
    assert np.allclose(auc_binary([0, 1], [0.5, 0.5]), 0.5)
    try:
        auc_binary([1, 1], [0.2, 0.3])
        assert False
    except ValueError:
        pass

test_auc_binary()
print("All tests passed.")
```

## STOP HERE

请先完成带提示练习区和无提示练习区，再查看参考答案。

## 参考答案与解析

```python
import numpy as np

def auc_binary(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires at least one positive and one negative sample")
    total = 0.0
    for p in pos:
        total += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return total / (len(pos) * len(neg))
```

### 解析

1. AUC 不依赖阈值，只看排序关系。
2. 平分 ties 是常见约定。
3. 如果全是正样本或全是负样本，AUC 无定义。

## 工程要点 / 面试追问

### 核心公式

- $AUC=\int_0^1 TPR(FPR)\,dFPR$。
- 二分类排序视角：$AUC=P(s^+>s^-)+0.5P(s^+=s^-)$。

### 易错点

- 把准确率当成 AUC；AUC 衡量排序能力，不依赖某个固定阈值。
- 忘记处理相等分数，ties 通常按 0.5 计。
- 只有一个类别时 ROC-AUC 没有定义。
- AUC 高不代表概率校准好，也不代表业务阈值下 precision 一定高。

### 面试追问

- 为什么 ROC-AUC 在类别不均衡时仍常被使用？
- PR-AUC 和 ROC-AUC 的差别是什么，极端不均衡时更关注哪个？
- AUC 的排序概率解释是什么？
- 如果预测分数做单调变换，AUC 会变吗？

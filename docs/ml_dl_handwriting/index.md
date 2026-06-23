# 01 ML/DL Handwriting

这个文件夹用于沉淀机器学习、深度学习和 LLM 训练相关的手撕模块。

## 初始范围

第一阶段只做：

- Math / Primitive
- Classical ML
- Deep Learning Basics
- Attention / Transformer
- LLM Training / Alignment

第一阶段暂不做：

- CV / Detection
- Recommendation

### Machine Learning

- KNN
- KMeans
- Linear Regression
- Logistic Regression
- Naive Bayes
- Decision Tree
- PCA

### Deep Learning

- MLP forward/backward
- Softmax and cross entropy
- BatchNorm / LayerNorm
- Optimizers: SGD, Momentum, Adam
- Attention
- Transformer block

### LLM Training

- Tokenization basics
- Causal LM loss
- SFT data collator
- LoRA adapter
- DPO loss
- Reward model pairwise loss

## Source Policy

题源可以参考公开面试题、自己整理的题意、以及开源项目中的可复现片段。不要直接复制受版权保护的整题文本；优先写成自己的抽象题面和测试。

## Implementation Constraint

默认手写约束：

- 允许 Python 基础语法、`list`、`dict`、`math`。
- 允许 NumPy 做矩阵计算。
- 不允许调用 sklearn、scipy、torch、transformers 里的现成算法、层、损失或指标。
- 如果某个模块必须使用 torch 才能贴近项目源码，需要在模块开头单独说明。

题型索引见：

- `source-research/niuke-ml-dl-topic-index.md`

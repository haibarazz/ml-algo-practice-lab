# 面试追问与公式参考来源

本文件记录手撕模块「工程要点 / 面试追问」的公开资料来源。`source-research/` 不会被同步到 VitePress 站点，学习页面不展示这些内部线索。

## 面试追问题型来源

- [amitshekhariitbhu/machine-learning-interview-questions](https://github.com/amitshekhariitbhu/machine-learning-interview-questions)：用于整理 ML 常见八股题型，如 loss、dropout、MLP、cross entropy、logits、指标等。
- [Data-Science-Interview-Questions-Answers: Deep Learning Questions](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/Deep%20Learning%20Questions%20%26%20Answers%20for%20Data%20Scientists.md)：用于整理深度学习常见追问，如 BatchNorm、Dropout、optimizer、momentum 等。
- [AIML.com Deep Learning interview topics](https://aiml.com/category/ml-interview-questions/deep-learning/)：用于补充 LayerNorm vs BatchNorm、activation、Transformer、positional encoding、perplexity、SFT 等题型覆盖。
- [GeeksforGeeks Deep Learning Interview Questions](https://www.geeksforgeeks.org/deep-learning/deep-learning-interview-questions/)：用于补充梯度消失/爆炸、梯度裁剪、Adam/RMSProp 等训练稳定性追问。
- [InterviewCoder ML interview questions](https://www.interviewcoder.co/blog/ml-interview-questions)：用于补充 logistic regression、multiclass、metrics、class imbalance 等面试问法。

## 公式与工程语义校准来源

- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)、[mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)、[roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)：用于校准 MSE、AUC、分类指标相关说法。
- [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)、[DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)：用于校准 KMeans objective、DBSCAN core/noise/eps 相关说法。
- [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)、[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)、[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)、[GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)：用于校准经典 ML 模块的边界条件和工程问法。
- [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)、[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)、[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)、[BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)、[LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)、[RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)、[Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)、[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：用于校准深度学习算子语义、shape、数值稳定性和训练/推理差异。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：用于校准 scaled dot-product attention、multi-head attention、sinusoidal positional encoding。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)：用于校准 Adam 一阶/二阶矩、bias correction 和自适应步长说法。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：用于校准 RMSNorm 与 LayerNorm 的差异。
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)：用于校准 RoPE 的旋转位置编码和相对位置信息说法。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)：用于校准 SwiGLU / GeGLU / gated FFN 相关说法。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) 与 [OpenReview 条目](https://openreview.net/forum?id=HPuSIXJaa9)：用于校准 DPO loss、reference model、beta 和偏好优化语义。

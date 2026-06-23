# 牛客 ML/DL 手撕题型索引

本文件记录公开网页中出现过的机器学习、深度学习和大模型算法手撕考点。它只作为题源线索，不收录原题全文；进入正式模块前必须改写成自己的题面、样例和测试。

## 实现约束

默认约束：

- 允许 Python 基础语法、`list`、`dict`、`math`。
- 允许 NumPy 作为矩阵计算工具。
- 不允许直接调用 sklearn、scipy、torch/torchvision/transformers 中的现成算法、层、损失或指标实现。
- 深度学习模块如果需要张量语义，优先用 NumPy 复现；只有项目拆解模块需要贴近源项目时，才单独声明可用 torch。

## 来源线索

- 牛客《算法岗手撕练习》提到 ML/DL 可用 numpy 或 torch 实现，列出线性回归、MLP、Multi Head Attention、绝对位置编码、RoPE、交叉熵、AUC、BatchNorm、LayerNorm、KMeans、IoU、NMS 等考点。
- 牛客《总结:大模型算法面试手撕代码》列出位置编码、MHA/MQA/GQA/MLA、LayerNorm/RMSNorm/BatchNorm、FFN/ReLU/GeLU/SwiGLU、交叉熵、预训练损失、对比学习损失、熵、KL 散度等考点。
- 牛客《秋招社招算法面试基础算法手撕九道》列出 softmax、MSE、交叉熵、单头注意力、多头注意力、自注意力、交叉注意力、LayerNorm、AUC。
- 牛客《机器学习/算法校招面试考点汇总》覆盖 LR、SVM、朴素贝叶斯、距离计算、KMeans、DBSCAN、协同过滤、FM、GBDT/XGBoost、HMM、EM、PCA、AUC、softmax 等机器学习考点。

## 第一阶段高优先级模块

### 0. Primitive / Math

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| euclidean_distance_matrix | 两个矩阵之间的两两欧氏距离 | NumPy，禁止 sklearn |
| softmax_stable | 数值稳定 softmax | list + math / NumPy |
| cross_entropy | 多分类交叉熵 | NumPy |
| mse_loss | 均方误差 | list / NumPy |
| auc_binary | 二分类 AUC | list，排序实现 |
| entropy_kl | 熵与 KL 散度 | NumPy |
| pca_first_component | PCA 第一主成分 | NumPy，手写中心化和协方差 |

### 1. Classical ML

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| knn_classifier | KNN 距离、投票、tie-break | list / NumPy |
| kmeans_one_step | KMeans assignment + update | NumPy |
| kmeans_full | KMeans 迭代、收敛、空簇处理 | NumPy |
| linear_regression_gd | 线性回归 MSE + 梯度下降 | NumPy |
| logistic_regression_binary | sigmoid、BCE、梯度下降 | NumPy |
| logistic_regression_multiclass | softmax regression | NumPy |
| naive_bayes_text | 朴素贝叶斯文本分类 | list / dict |
| decision_tree_id3 | 信息增益选特征 | list / dict |
| dbscan_core | core point、density reachability | list / NumPy |
| em_gmm_one_step | GMM 的 E-step / M-step | NumPy |
| svm_hinge_loss | hinge loss 与 margin | NumPy |

### 2. Deep Learning Basics

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| mlp_forward | MLP 前向传播 | NumPy |
| mlp_backward | 两层 MLP 反向传播 | NumPy |
| relu_sigmoid_tanh | 激活函数及导数 | NumPy |
| batch_norm_forward | BatchNorm 训练态 forward | NumPy |
| batch_norm_backward | BatchNorm backward | NumPy |
| layer_norm_forward | LayerNorm forward | NumPy |
| rms_norm_forward | RMSNorm forward | NumPy |
| dropout_train_eval | dropout 训练/推理差异 | NumPy |
| adam_optimizer | Adam 参数更新 | NumPy |

### 3. Attention / Transformer

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| scaled_dot_product_attention | QK^T / sqrt(d)、mask、softmax | NumPy |
| self_attention | 自注意力封装 | NumPy |
| cross_attention | query 与 encoder outputs | NumPy |
| multi_head_attention | split heads、concat、output projection | NumPy |
| mha_with_mask | causal mask / padding mask | NumPy |
| sinusoidal_position_encoding | 绝对位置编码 | NumPy |
| rope_apply | RoPE 旋转位置编码 | NumPy |
| gqa_mqa_shapes | MQA/GQA 的 KV head 复用 | NumPy |
| ffn_relu_gelu | Transformer FFN | NumPy |
| swiglu | SwiGLU forward | NumPy |

### 4. LLM Training / Alignment

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| causal_lm_loss | next-token shift 与交叉熵 | NumPy |
| sft_label_mask | prompt mask、只训练 answer token | list / NumPy |
| preference_pair_format | chosen/rejected 数据组织 | list / dict |
| dpo_loss | DPO loss 核心公式 | NumPy |
| reward_pairwise_loss | pairwise reward loss | NumPy |
| contrastive_loss | InfoNCE / 对比学习损失 | NumPy |

### 5. CV / Detection Optional

第一阶段暂不做。保留索引用于后续扩展。

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| iou_boxes | bounding box IoU | list / NumPy |
| nms | 非极大值抑制 | list / NumPy |
| conv2d_naive | 2D 卷积朴素实现 | NumPy |
| max_pool2d_naive | pooling 朴素实现 | NumPy |

### 6. Recommendation Optional

第一阶段暂不做。保留索引用于后续扩展。

| Module | 考点 | 建议实现约束 |
| --- | --- | --- |
| user_cf | user-based collaborative filtering | list / NumPy |
| item_cf | item-based collaborative filtering | list / NumPy |
| matrix_factorization_sgd | MF 的 SGD 更新 | NumPy |
| fm_forward | Factorization Machine forward | NumPy |

## 进入正式模块的规则

每个索引条目进入 `ml_dl_handwriting/` 前，都要改写成：

1. 原理最小说明
2. 带提示练习区
3. 无提示练习区
4. 测试区
5. `STOP HERE`
6. 参考答案与解析

每个模块必须声明：

- 允许使用的工具：`list` / `numpy`
- 禁止使用的包：例如 sklearn、torch 内置层、现成 loss 或 metric
- 输入输出 shape
- 至少两个测试样例
- 常见错误

## References

- https://www.nowcoder.com/feed/main/detail/347eadbc31e942b6b84ccce81d68754b
- https://www.nowcoder.com/feed/main/detail/5a43a5d03e5746e5b743bb488a5c00f9
- https://www.nowcoder.com/discuss/651759346887503872
- https://www.nowcoder.com/discuss/353154899112304640

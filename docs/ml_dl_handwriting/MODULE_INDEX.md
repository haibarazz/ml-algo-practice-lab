# ML/DL Handwriting Modules

第一阶段模块目录。

## [Math / Primitive](./00_math_primitives/README.md)

- [euclidean_distance_matrix](./00_math_primitives/euclidean_distance_matrix/euclidean_distance_matrix.md): 两组向量之间的两两欧氏距离矩阵
- [softmax_stable](./00_math_primitives/softmax_stable/softmax_stable.md): 数值稳定 softmax
- [cross_entropy](./00_math_primitives/cross_entropy/cross_entropy.md): 多分类交叉熵
- [mse_loss](./00_math_primitives/mse_loss/mse_loss.md): 均方误差
- [auc_binary](./00_math_primitives/auc_binary/auc_binary.md): 二分类 AUC
- [entropy_kl](./00_math_primitives/entropy_kl/entropy_kl.md): 熵与 KL 散度
- [pca_first_component](./00_math_primitives/pca_first_component/pca_first_component.md): PCA 第一主成分

## [Classical ML](./01_classical_ml/README.md)

- [knn_classifier](./01_classical_ml/knn_classifier/knn_classifier.md): KNN 距离、投票与 tie-break
- [kmeans_one_step](./01_classical_ml/kmeans_one_step/kmeans_one_step.md): KMeans 一轮 assignment + update
- [kmeans_full](./01_classical_ml/kmeans_full/kmeans_full.md): KMeans 迭代、收敛与空簇处理
- [linear_regression_gd](./01_classical_ml/linear_regression_gd/linear_regression_gd.md): 线性回归 MSE 与梯度下降
- [logistic_regression_binary](./01_classical_ml/logistic_regression_binary/logistic_regression_binary.md): 二分类逻辑回归
- [logistic_regression_multiclass](./01_classical_ml/logistic_regression_multiclass/logistic_regression_multiclass.md): 多分类 softmax regression
- [naive_bayes_text](./01_classical_ml/naive_bayes_text/naive_bayes_text.md): 朴素贝叶斯文本分类
- [decision_tree_id3](./01_classical_ml/decision_tree_id3/decision_tree_id3.md): ID3 信息增益选特征
- [dbscan_core](./01_classical_ml/dbscan_core/dbscan_core.md): DBSCAN 核心点与密度可达
- [em_gmm_one_step](./01_classical_ml/em_gmm_one_step/em_gmm_one_step.md): GMM 的 E-step / M-step
- [svm_hinge_loss](./01_classical_ml/svm_hinge_loss/svm_hinge_loss.md): SVM hinge loss 与 margin

## [Deep Learning Basics](./02_dl_basics/README.md)

- [mlp_forward](./02_dl_basics/mlp_forward/mlp_forward.md): MLP 前向传播
- [mlp_backward](./02_dl_basics/mlp_backward/mlp_backward.md): 两层 MLP 反向传播
- [relu_sigmoid_tanh](./02_dl_basics/relu_sigmoid_tanh/relu_sigmoid_tanh.md): 激活函数及导数
- [batch_norm_forward](./02_dl_basics/batch_norm_forward/batch_norm_forward.md): BatchNorm 训练态 forward
- [batch_norm_backward](./02_dl_basics/batch_norm_backward/batch_norm_backward.md): BatchNorm backward
- [layer_norm_forward](./02_dl_basics/layer_norm_forward/layer_norm_forward.md): LayerNorm forward
- [rms_norm_forward](./02_dl_basics/rms_norm_forward/rms_norm_forward.md): RMSNorm forward
- [dropout_train_eval](./02_dl_basics/dropout_train_eval/dropout_train_eval.md): Dropout 训练/推理差异
- [adam_optimizer](./02_dl_basics/adam_optimizer/adam_optimizer.md): Adam 参数更新

## [Attention / Transformer](./03_attention_transformer/README.md)

- [scaled_dot_product_attention](./03_attention_transformer/scaled_dot_product_attention/scaled_dot_product_attention.md): Scaled Dot-Product Attention
- [self_attention](./03_attention_transformer/self_attention/self_attention.md): Self-Attention 封装
- [cross_attention](./03_attention_transformer/cross_attention/cross_attention.md): Cross-Attention
- [multi_head_attention](./03_attention_transformer/multi_head_attention/multi_head_attention.md): Multi-Head Attention
- [mha_with_mask](./03_attention_transformer/mha_with_mask/mha_with_mask.md): MHA with causal / padding mask
- [sinusoidal_position_encoding](./03_attention_transformer/sinusoidal_position_encoding/sinusoidal_position_encoding.md): 正弦绝对位置编码
- [rope_apply](./03_attention_transformer/rope_apply/rope_apply.md): RoPE 旋转位置编码
- [gqa_mqa_shapes](./03_attention_transformer/gqa_mqa_shapes/gqa_mqa_shapes.md): MQA / GQA 的 KV head 复用
- [ffn_relu_gelu](./03_attention_transformer/ffn_relu_gelu/ffn_relu_gelu.md): Transformer FFN with ReLU / GeLU
- [swiglu](./03_attention_transformer/swiglu/swiglu.md): SwiGLU forward

## [LLM Training / Alignment](./04_llm_training_alignment/README.md)

- [causal_lm_loss](./04_llm_training_alignment/causal_lm_loss/causal_lm_loss.md): Causal LM next-token loss
- [sft_label_mask](./04_llm_training_alignment/sft_label_mask/sft_label_mask.md): SFT label mask
- [preference_pair_format](./04_llm_training_alignment/preference_pair_format/preference_pair_format.md): chosen / rejected 偏好数据组织
- [dpo_loss](./04_llm_training_alignment/dpo_loss/dpo_loss.md): DPO loss
- [reward_pairwise_loss](./04_llm_training_alignment/reward_pairwise_loss/reward_pairwise_loss.md): Pairwise reward loss
- [contrastive_loss](./04_llm_training_alignment/contrastive_loss/contrastive_loss.md): InfoNCE / 对比学习损失

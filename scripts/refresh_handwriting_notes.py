#!/usr/bin/env python3
"""Refresh ML/DL handwriting notes and inline them into module pages."""

from __future__ import annotations

from pathlib import Path

from handwriting_interview_answers import MODULE_ANSWERS


ROOT = Path(__file__).resolve().parents[1]


MODULE_NOTES: dict[str, dict[str, list[str] | str]] = {
    "ml_dl_handwriting/00_math_primitives/euclidean_distance_matrix": {
        "title": "Euclidean Distance Matrix Notes",
        "formula": [
            r"$D_{ij}=\lVert x_i-y_j\rVert_2$；实现时常先算平方距离：$\lVert x_i\rVert^2+\lVert y_j\rVert^2-2x_i^\top y_j$。",
            r"如果只比较最近邻，平方距离和真实距离排序一致，可以省掉 $\sqrt{\cdot}$。"
        ],
        "pitfalls": [
            "把输出 shape 写反；给定 X shape 为 `[m,d]`、Y shape 为 `[n,d]` 时，结果应为 `[m,n]`。",
            "显式构造 `[m,n,d]` 中间张量会占用大量内存，大数据时优先使用矩阵乘法展开式。",
            "浮点误差可能让平方距离出现极小负数，开方前可做 `clip(min=0)`。",
            "特征尺度差异会直接影响欧氏距离，KNN/KMeans 前通常要考虑标准化。"
        ],
        "questions": [
            "如何用矩阵乘法实现两两欧氏距离？时间复杂度和内存复杂度分别是多少？",
            "为什么最近邻检索时常比较平方距离而不是距离本身？",
            "余弦相似度和欧氏距离分别适合什么特征表示？",
            "如果样本量很大，如何分块计算距离矩阵以避免内存爆掉？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/softmax_stable": {
        "title": "Softmax Stable Notes",
        "formula": [
            r"$\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$。",
            r"数值稳定写法：$\operatorname{softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}$。"
        ],
        "pitfalls": [
            "忘记减最大值，遇到大 logits 会 overflow。",
            "`axis` 写死成最后一维，批量输入时不够通用。",
            "没有 `keepdims=True`，二维或高维输入广播容易错。",
            "softmax 输出不是独立概率；所有类别概率相加必须为 1。"
        ],
        "questions": [
            "为什么减去同一个常数不会改变 softmax 结果？",
            "softmax 和 sigmoid 在二分类、多分类场景下有什么关系？",
            "为什么交叉熵里常把 log-softmax 和 NLL 合并计算？",
            "temperature 对 softmax 分布的尖锐程度有什么影响？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/cross_entropy": {
        "title": "Cross Entropy Notes",
        "formula": [
            r"one-hot target 下：$L=-\sum_c y_c\log p_c$；若 target 是类别下标 $t$，则 $L=-\log p_t$。",
            r"对 logits 使用 softmax + CE 时，常见梯度为 $\frac{\partial L}{\partial z}=p-y$。"
        ],
        "pitfalls": [
            "先 softmax 再 log 容易数值不稳定，应使用 log-sum-exp 或 fused cross entropy。",
            "target 下标和 one-hot 表示混用，导致 shape 或语义错误。",
            "忘记 batch mean，使 loss 尺度随 batch size 变化。",
            "把多分类单标签 CE 和多标签 BCE 混淆。"
        ],
        "questions": [
            "softmax + cross entropy 的梯度为什么可以化简成 `p - y`？",
            "为什么 PyTorch 的 `CrossEntropyLoss` 输入通常是 logits 而不是概率？",
            "label smoothing 会怎样改变 target 分布和梯度？",
            "类别不均衡时，cross entropy 可以怎样加权？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/mse_loss": {
        "title": "MSE Loss Notes",
        "formula": [
            r"$MSE=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$。",
            r"若对预测值求梯度：$\frac{\partial MSE}{\partial \hat y_i}=\frac{2}{N}(\hat y_i-y_i)$。"
        ],
        "pitfalls": [
            "sum 和 mean 混用会导致 loss 与梯度尺度不一致。",
            "`y_true - y_pred` 和 `y_pred - y_true` 对 loss 无影响，但对梯度方向有影响。",
            "多维输出时要说清楚是按所有元素平均，还是先按样本再按输出维聚合。",
            "MSE 对异常值敏感，数据噪声重尾时不一定是好选择。"
        ],
        "questions": [
            "MSE 为什么比 MAE 更容易受异常值影响？",
            "MSE、RMSE、MAE 分别适合什么回归评价场景？",
            "如果训练时用 MSE，梯度分母应该除以 batch size 还是元素总数？",
            "Huber loss 如何在 MSE 和 MAE 之间折中？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/auc_binary": {
        "title": "Binary AUC Notes",
        "formula": [
            r"$AUC=\int_0^1 TPR(FPR)\,dFPR$。",
            r"二分类排序视角：$AUC=P(s^+>s^-)+0.5P(s^+=s^-)$。"
        ],
        "pitfalls": [
            "把准确率当成 AUC；AUC 衡量排序能力，不依赖某个固定阈值。",
            "忘记处理相等分数，ties 通常按 0.5 计。",
            "只有一个类别时 ROC-AUC 没有定义。",
            "AUC 高不代表概率校准好，也不代表业务阈值下 precision 一定高。"
        ],
        "questions": [
            "为什么 ROC-AUC 在类别不均衡时仍常被使用？",
            "PR-AUC 和 ROC-AUC 的差别是什么，极端不均衡时更关注哪个？",
            "AUC 的排序概率解释是什么？",
            "如果预测分数做单调变换，AUC 会变吗？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/entropy_kl": {
        "title": "Entropy And KL Notes",
        "formula": [
            r"$H(p)=-\sum_i p_i\log p_i$。",
            r"$D_{KL}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}$，且 $H(p,q)=H(p)+D_{KL}(p\Vert q)$。"
        ],
        "pitfalls": [
            "把 KL 当成距离使用，但 KL 不满足对称性，也不满足三角不等式。",
            "忘记归一化概率，导致熵和 KL 没有概率意义。",
            "`0 log 0` 的极限是 0，但直接计算会得到 `nan`。",
            "KL 的方向很重要，`KL(p||q)` 和 `KL(q||p)` 对模式覆盖的偏好不同。"
        ],
        "questions": [
            "交叉熵、熵和 KL 散度之间是什么关系？",
            "为什么知识蒸馏和策略约束里经常出现 KL？",
            "KL 为什么非负？什么时候等于 0？",
            "正向 KL 和反向 KL 在分布拟合上的直觉差异是什么？"
        ],
    },
    "ml_dl_handwriting/00_math_primitives/pca_first_component": {
        "title": "PCA First Component Notes",
        "formula": [
            r"中心化后协方差矩阵 $C=\frac{1}{n}X_c^\top X_c$。",
            r"第一主成分 $v_1=\arg\max_{\lVert v\rVert=1} v^\top C v$，对应最大特征值的特征向量。"
        ],
        "pitfalls": [
            "忘记中心化，第一主成分会被均值偏移污染。",
            "直接用 `eig` 可能出现很小的复数数值噪声，对称矩阵优先用 `eigh` 或 SVD。",
            "特征向量正负号不唯一，测试时应比较方向等价而不是固定符号。",
            "不同特征量纲差异大时，是否标准化会显著影响 PCA。"
        ],
        "questions": [
            "为什么第一主成分对应协方差矩阵最大特征值？",
            "PCA 和 SVD 的关系是什么？",
            "PCA 是有监督还是无监督方法？它会不会利用标签？",
            "PCA 降维后保留多少维通常怎么选？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/knn_classifier": {
        "title": "KNN Classifier Notes",
        "formula": [
            r"$\hat y=\operatorname{mode}\{y_i: x_i \in N_k(x)\}$。",
            r"常见距离：欧氏距离 $\lVert x-x_i\rVert_2$，也可按任务换成曼哈顿距离或余弦距离。"
        ],
        "pitfalls": [
            "`k` 大于训练样本数没有处理。",
            "投票 tie-break 不确定，复现实验时必须固定规则。",
            "特征尺度不同会显著影响距离，通常要做标准化。",
            "KNN 训练很轻，但预测要扫训练集，线上延迟可能很高。"
        ],
        "questions": [
            "KNN 的训练复杂度和预测复杂度分别是多少？",
            "如何选择 K？K 太大或太小分别有什么问题？",
            "如何加速最近邻检索，比如 KDTree、BallTree、ANN？",
            "KNN 为什么容易受维度灾难影响？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/kmeans_one_step": {
        "title": "KMeans One Step Notes",
        "formula": [
            r"assignment：$z_i=\arg\min_k\lVert x_i-\mu_k\rVert^2$。",
            r"update：$\mu_k=\frac{1}{|C_k|}\sum_{i:z_i=k}x_i$。"
        ],
        "pitfalls": [
            "距离矩阵 shape 写反，导致样本维和中心维混淆。",
            "assignment 只需要比较平方距离，不必开根号。",
            "空簇需要明确策略：保持旧中心、重采样或选择最远点。",
            "`np.argmin` 默认返回第一个最小值，tie-break 要说明。"
        ],
        "questions": [
            "KMeans 一轮迭代为什么分为 assignment 和 update？",
            "KMeans 对初始中心敏感吗？KMeans++ 解决了什么问题？",
            "K 怎么选？肘部法和 silhouette score 的直觉是什么？",
            "如果数据量很大，如何加速 assignment 或做 mini-batch KMeans？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/kmeans_full": {
        "title": "KMeans Full Notes",
        "formula": [
            r"目标函数：$\min_{\{C_k\},\{\mu_k\}}\sum_k\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2$。",
            r"每轮 assignment/update 不增加该目标，但只能保证收敛到局部最优。"
        ],
        "pitfalls": [
            "收敛判断放错位置，导致多跑一轮或提前停止。",
            "空簇导致 `mean of empty slice`，必须定义处理策略。",
            "初始化不同结果不同，工程中常跑多次取 inertia 最小的结果。",
            "没有设置 `max_iter`，异常数据上可能运行过久。"
        ],
        "questions": [
            "KMeans 为什么只能保证局部最优？",
            "KMeans++ 的初始化思想是什么？",
            "KMeans 适合非凸簇或不同密度簇吗？为什么？",
            "KMeans、GMM、DBSCAN 的核心假设有什么不同？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/linear_regression_gd": {
        "title": "Linear Regression GD Notes",
        "formula": [
            r"$\hat y=Xw+b$，$L=\frac{1}{n}\sum_i(\hat y_i-y_i)^2$。",
            r"$\nabla_w L=\frac{2}{n}X^\top(Xw+b-y)$，$\nabla_b L=\frac{2}{n}\sum_i(\hat y_i-y_i)$。"
        ],
        "pitfalls": [
            "梯度忘记除以 batch size，导致学习率含义变化。",
            "bias 梯度写成向量而不是标量或按输出维聚合。",
            "学习率过大导致发散，过小导致收敛慢。",
            "没有检查输入是否需要加截距项，闭式解和 GD 实现容易不一致。"
        ],
        "questions": [
            "线性回归的闭式解是什么？什么时候不适合直接用闭式解？",
            "GD、SGD、mini-batch SGD 的区别是什么？",
            "为什么特征标准化会影响梯度下降收敛速度？",
            "L1/L2 正则会怎样改变目标函数和解的性质？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/logistic_regression_binary": {
        "title": "Binary Logistic Regression Notes",
        "formula": [
            r"$p(y=1|x)=\sigma(w^\top x+b)$，$\sigma(z)=\frac{1}{1+e^{-z}}$。",
            r"$L=-[y\log p+(1-y)\log(1-p)]$，对 logits 的梯度为 $p-y$。"
        ],
        "pitfalls": [
            "把 logistic regression 和线性回归混淆；LR 输出的是类别概率。",
            "sigmoid 直接计算可能溢出，BCE with logits 更稳定。",
            "BCE 梯度符号写反，参数更新方向会错。",
            "默认阈值 0.5 不一定适合类别不均衡或不同业务成本。"
        ],
        "questions": [
            "逻辑回归为什么是判别模型？",
            "逻辑回归和线性回归的目标函数有什么区别？",
            "为什么 BCE with logits 比 sigmoid 后再 BCE 更稳定？",
            "二分类阈值如何根据 precision/recall 或业务成本调整？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/logistic_regression_multiclass": {
        "title": "Multiclass Logistic Regression Notes",
        "formula": [
            r"$p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}$，$z=XW+b$。",
            r"$L=-\log p_{y}$，对 logits 的梯度为 $p-\operatorname{onehot}(y)$。"
        ],
        "pitfalls": [
            "忘记稳定 softmax，logits 较大时容易 overflow。",
            "`grad = probs` 后原地修改，后续再用 `probs` 会被污染。",
            "W shape 写成 `[C,D]` 或 `[D,C]` 时要全程保持一致。",
            "one-vs-rest 和 multinomial softmax 的训练目标不同，不要混用结论。"
        ],
        "questions": [
            "softmax regression 和 one-vs-rest logistic regression 有什么区别？",
            "cross entropy 梯度为什么能化简为 `probs - onehot`？",
            "多分类类别不均衡时可以怎样加权？",
            "softmax 输出概率是否一定校准？如果不校准怎么办？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/naive_bayes_text": {
        "title": "Naive Bayes Text Notes",
        "formula": [
            r"$\hat y=\arg\max_c \log P(c)+\sum_j x_j\log P(w_j|c)$。",
            r"Laplace 平滑：$P(w|c)=\frac{count(w,c)+\alpha}{\sum_v count(v,c)+\alpha |V|}$。"
        ],
        "pitfalls": [
            "概率直接相乘会下溢，应在 log 空间相加。",
            "未见词概率为 0，需要平滑。",
            "类别先验忘记加入，类别不均衡时影响明显。",
            "多项式 NB、伯努利 NB、Gaussian NB 的特征假设不同。"
        ],
        "questions": [
            "朴素贝叶斯“朴素”在哪里？独立性假设为什么仍常有效？",
            "多项式 NB 和伯努利 NB 的区别是什么？",
            "为什么文本分类里常在 log 空间计算？",
            "平滑系数 alpha 太大或太小会产生什么影响？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/decision_tree_id3": {
        "title": "Decision Tree ID3 Notes",
        "formula": [
            r"$H(D)=-\sum_c p_c\log p_c$。",
            r"$Gain(D,A)=H(D)-\sum_v\frac{|D_v|}{|D|}H(D_v)$。"
        ],
        "pitfalls": [
            "条件熵忘记按子集样本占比加权。",
            "log 底数不影响特征排序，但会影响信息增益数值。",
            "连续特征需要额外找切分点，本模块只处理离散特征。",
            "ID3 偏好多取值特征，工程中常用增益率或正则化限制。"
        ],
        "questions": [
            "ID3、C4.5、CART 的划分准则有什么区别？",
            "信息增益为什么偏好多取值特征？",
            "决策树如何防止过拟合？预剪枝和后剪枝有什么区别？",
            "Gini impurity 和 entropy 的直觉差异是什么？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/dbscan_core": {
        "title": "DBSCAN Core Notes",
        "formula": [
            r"核心点：$|N_\varepsilon(x)|\ge minPts$，其中 $N_\varepsilon(x)=\{y:\operatorname{dist}(x,y)\le\varepsilon\}$。",
            "密度可达由核心点邻域扩展得到，无法从任何核心点扩展到的点可标为噪声。"
        ],
        "pitfalls": [
            "`min_samples` 是否包含自身要说明，本项目约定包含自身。",
            "噪声点后续可能变成边界点，不能过早永久丢弃。",
            "`eps` 对尺度敏感，输入特征通常需要标准化。",
            "不同密度簇上 DBSCAN 可能表现不好，一个全局 eps 难以兼顾。"
        ],
        "questions": [
            "DBSCAN 和 KMeans 的核心区别是什么？",
            "DBSCAN 为什么可以发现任意形状簇并识别噪声？",
            "eps 和 min_samples 应该如何选择？",
            "DBSCAN 为什么不天然支持对新样本直接 predict？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/em_gmm_one_step": {
        "title": "EM GMM One Step Notes",
        "formula": [
            r"E-step：$\gamma_{ik}=\frac{\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j\pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}$。",
            r"M-step：$N_k=\sum_i\gamma_{ik}$，$\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i$，$\pi_k=N_k/n$。"
        ],
        "pitfalls": [
            "E-step 应按每个样本在所有簇上的责任度归一化。",
            "M-step 更新方差时要使用新均值，而不是旧均值。",
            "方差过小会导致数值不稳定，工程中常加最小方差或协方差正则。",
            "概率密度连乘容易下溢，高维实现常用 log-sum-exp。"
        ],
        "questions": [
            "EM 为什么能单调不降低数据似然？",
            "GMM 和 KMeans 的关系是什么？什么时候 GMM 会退化得像 KMeans？",
            "GMM 中协方差矩阵 full、diag、spherical 有什么差别？",
            "EM 对初始化敏感吗？如何缓解局部最优？"
        ],
    },
    "ml_dl_handwriting/01_classical_ml/svm_hinge_loss": {
        "title": "SVM Hinge Loss Notes",
        "formula": [
            r"二分类 hinge loss：$L=\max(0,1-y(w^\top x+b))$，其中 $y\in\{-1,+1\}$。",
            r"带 L2 正则的软间隔目标常写为 $\frac{1}{2}\lVert w\rVert^2+C\sum_i\xi_i$。"
        ],
        "pitfalls": [
            "标签用 0/1 而不是 -1/+1，会让 margin 公式失效。",
            "忘记正则项，无法体现最大间隔。",
            "active mask 条件写反；只有 margin 小于 1 的样本贡献 hinge 梯度。",
            "hinge loss 在折点不可导，通常使用次梯度。"
        ],
        "questions": [
            "SVM 的 margin 几何意义是什么？",
            "hinge loss 和 logistic loss 的差别是什么？",
            "C 参数变大或变小会怎样影响间隔和误分类？",
            "核 SVM 为什么在大样本上训练代价高？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/mlp_forward": {
        "title": "MLP Forward Notes",
        "formula": [
            r"两层 MLP：$h=\phi(XW_1+b_1)$，$\hat y=hW_2+b_2$。",
            r"如果 $\phi$ 是 ReLU，则 $\phi(z)=\max(0,z)$。"
        ],
        "pitfalls": [
            "bias 广播维度不清楚，导致单样本和 batch 输入表现不一致。",
            "把 ReLU 放在第二层之后会改变题目定义。",
            "不缓存中间值，后续 backward 难写。",
            "多层线性层如果不加激活，本质仍等价于一个线性层。"
        ],
        "questions": [
            "为什么 MLP 需要非线性激活函数？",
            "多层线性层不加激活等价于什么？",
            "隐藏层宽度会怎样影响表达能力和过拟合风险？",
            "forward 中需要缓存哪些变量给 backward 使用？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/mlp_backward": {
        "title": "MLP Backward Notes",
        "formula": [
            r"链式法则：$\frac{\partial L}{\partial W_2}=h^\top\frac{\partial L}{\partial \hat y}$，$\frac{\partial L}{\partial h}=\frac{\partial L}{\partial \hat y}W_2^\top$。",
            r"ReLU backward：$\frac{\partial h}{\partial z_1}=\mathbf{1}[z_1>0]$。"
        ],
        "pitfalls": [
            "MSE 梯度分母用 batch size 还是元素总数，要和 loss 定义一致。",
            "ReLU backward 推荐缓存 `z1`，不要依赖被后续修改过的激活值。",
            "`db` 忘记沿 batch 维求和。",
            "矩阵乘法方向写反，导致 shape 对不上或梯度转置。"
        ],
        "questions": [
            "如果输出层改成 softmax + cross entropy，最后一层梯度如何化简？",
            "为什么需要激活函数？没有激活时 backward 会发生什么？",
            "梯度消失和梯度爆炸分别可能出现在哪里？",
            "如何用数值梯度检查 backward 是否正确？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/relu_sigmoid_tanh": {
        "title": "Activation Notes",
        "formula": [
            r"ReLU：$f(x)=\max(0,x)$；Sigmoid：$\sigma(x)=1/(1+e^{-x})$；Tanh：$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$。",
            r"导数：$\sigma'(x)=\sigma(x)(1-\sigma(x))$，$\tanh'(x)=1-\tanh^2(x)$。"
        ],
        "pitfalls": [
            "sigmoid 直接 `exp(-x)` 在大负数上可能溢出。",
            "ReLU 在 0 处不可导，需要说明实现约定。",
            "tanh 梯度公式应是 `1 - tanh(x)^2`，不是 `1 - x^2`。",
            "sigmoid/tanh 饱和区梯度很小，深层网络中容易梯度消失。"
        ],
        "questions": [
            "ReLU 为什么能缓解梯度消失？它有什么缺点？",
            "sigmoid、tanh、ReLU 的输出范围和梯度特点分别是什么？",
            "什么是 dead ReLU？如何缓解？",
            "GeLU/SiLU 相比 ReLU 的直觉优势是什么？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/batch_norm_forward": {
        "title": "BatchNorm Forward Notes",
        "formula": [
            r"$\mu_B=\frac{1}{m}\sum_i x_i$，$\sigma_B^2=\frac{1}{m}\sum_i(x_i-\mu_B)^2$。",
            r"$\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$，$y_i=\gamma\hat x_i+\beta$。"
        ],
        "pitfalls": [
            "把 LayerNorm 的 axis 用到 BatchNorm；BN 通常按 batch 维统计每个特征。",
            "忘记 eps，方差很小时会除零或放大噪声。",
            "训练态用 batch stats，推理态用 running stats，本模块只做训练态。",
            "batch size 很小时统计不稳定，效果可能变差。"
        ],
        "questions": [
            "BatchNorm 训练和推理有什么不同？",
            "BN 为什么对 batch size 敏感？",
            "BN 的 gamma 和 beta 有什么作用？",
            "为什么 Transformer/LLM 更常用 LayerNorm 或 RMSNorm？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/batch_norm_backward": {
        "title": "BatchNorm Backward Notes",
        "formula": [
            r"紧凑写法：$dx=\frac{\gamma}{m\sqrt{\sigma^2+\epsilon}}\left(m\,dy-\sum dy-\hat x\sum(dy\hat x)\right)$。",
            r"$d\gamma=\sum_i dy_i\hat x_i$，$d\beta=\sum_i dy_i$。"
        ],
        "pitfalls": [
            "求和 axis 写错，batch 维和 feature 维混淆。",
            "忘记乘 gamma，导致传回上一层的梯度尺度错误。",
            "backward 必须使用 forward 缓存的同一个 `var + eps`。",
            "手推链路较长，推荐先写数值梯度校验。"
        ],
        "questions": [
            "手推 BN backward 的关键链路是什么？",
            "为什么 `d_beta` 是上游梯度求和，`d_gamma` 要乘 normalized input？",
            "BN backward 中哪些量必须从 forward 缓存？",
            "BN 对 batch size 敏感会怎样影响训练稳定性？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/layer_norm_forward": {
        "title": "LayerNorm Forward Notes",
        "formula": [
            r"$\mu=\frac{1}{d}\sum_j x_j$，$\sigma^2=\frac{1}{d}\sum_j(x_j-\mu)^2$。",
            r"$y=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$，统计量通常在每个样本的特征维上计算。"
        ],
        "pitfalls": [
            "和 BatchNorm 混淆 axis；LN 不依赖 batch 维统计。",
            "忘记 `keepdims`，gamma/beta 广播容易错。",
            "normalized_shape 和最后若干维不匹配。",
            "Pre-LN/Post-LN 的位置改变会影响深层 Transformer 的梯度流。"
        ],
        "questions": [
            "LayerNorm 和 BatchNorm 的统计维度有什么区别？",
            "为什么 NLP/Transformer 中 LayerNorm 比 BatchNorm 常见？",
            "Pre-LN 和 Post-LN Transformer 的训练稳定性有什么差异？",
            "LayerNorm 的 gamma/beta 是否必须？去掉会怎样？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/rms_norm_forward": {
        "title": "RMSNorm Forward Notes",
        "formula": [
            r"$RMS(x)=\sqrt{\frac{1}{d}\sum_j x_j^2+\epsilon}$。",
            r"$y=\gamma\frac{x}{RMS(x)}$；RMSNorm 去掉了 LayerNorm 的减均值步骤。"
        ],
        "pitfalls": [
            "写成 LayerNorm，额外减了均值。",
            "沿 batch 维求 RMS，而不是沿最后的 feature 维。",
            "忘记 gamma 或 gamma shape 不能广播。",
            "RMSNorm 只保留 re-scaling，不提供 re-centering invariance。"
        ],
        "questions": [
            "RMSNorm 相比 LayerNorm 省掉了什么计算？",
            "为什么很多 LLM 使用 RMSNorm？",
            "RMSNorm 保留了什么不变性，又失去了什么不变性？",
            "RMSNorm 的 eps 应该放在哪里？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/dropout_train_eval": {
        "title": "Dropout Notes",
        "formula": [
            r"inverted dropout 训练态：$y=\frac{m\odot x}{1-p}$，$m\sim Bernoulli(1-p)$。",
            r"推理态通常直接 $y=x$。"
        ],
        "pitfalls": [
            "推理时还随机丢弃，导致输出不稳定。",
            "训练时没有除以 keep probability，导致期望变小。",
            "`p` 和 `keep_prob` 混淆。",
            "mask 需要和输入 shape 可广播，并且通常只在训练态采样。"
        ],
        "questions": [
            "Dropout 为什么能缓解过拟合？",
            "什么是 inverted dropout？为什么推理时不用再缩放？",
            "Dropout 和 BatchNorm 同时使用有什么注意点？",
            "Transformer 中 attention dropout 和 FFN dropout 分别作用在哪里？"
        ],
    },
    "ml_dl_handwriting/02_dl_basics/adam_optimizer": {
        "title": "Adam Optimizer Notes",
        "formula": [
            r"$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$，$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$。",
            r"$\hat m_t=m_t/(1-\beta_1^t)$，$\hat v_t=v_t/(1-\beta_2^t)$，$\theta_t=\theta_{t-1}-\alpha\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。"
        ],
        "pitfalls": [
            "忘记 bias correction，训练早期步长会偏小。",
            "`t` 从 0 开始带入校正项会除零。",
            "`v` 应累计 `grad ** 2`，不是累计 `grad`。",
            "Adam 的 weight decay 和 L2 regularization 在自适应优化器中并不完全等价，AdamW 会解耦。"
        ],
        "questions": [
            "Adam 和 SGD momentum 的区别是什么？",
            "bias correction 为什么必要？",
            "AdamW 为什么把 weight decay 解耦？",
            "Adam 在稀疏梯度、噪声梯度场景下有什么优势和风险？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/scaled_dot_product_attention": {
        "title": "Scaled Dot Product Attention Notes",
        "formula": [
            r"$Attention(Q,K,V)=softmax(\frac{QK^\top}{\sqrt{d_k}})V$。",
            r"mask 通常在 softmax 前加到 logits 上，被屏蔽位置加一个很大的负数。"
        ],
        "pitfalls": [
            r"忘记除以 $\sqrt{d_k}$，维度大时 logits 方差过大，softmax 容易饱和。",
            "mask 在 softmax 后处理会导致概率和不为 1。",
            "K 的转置维度写错，batch/head/seq 维容易混乱。",
            "全 mask 行可能产生 `nan`，工程上要定义处理策略。"
        ],
        "questions": [
            "为什么 dot-product attention 需要 scaling？",
            "causal mask 和 padding mask 分别解决什么问题？",
            "attention 的时间和显存复杂度是多少？",
            "为什么 softmax 前加大负数可以实现 mask？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/self_attention": {
        "title": "Self Attention Notes",
        "formula": [
            r"self-attention 中 $Q=XW_Q$，$K=XW_K$，$V=XW_V$，三者来自同一序列。",
            r"输出 $O=Attention(Q,K,V)$，长度通常等于 query 序列长度。"
        ],
        "pitfalls": [
            "Q/K/V shape 不一致，尤其是最后一维和 head 维。",
            "把 self-attention 和 cross-attention 混淆；self-attention 的 Q/K/V 来自同一输入。",
            "没有保留 batch 维，单样本测试通过但 batch 输入失败。",
            "忘记 mask 时，decoder 会看到未来 token。"
        ],
        "questions": [
            "self-attention 的时间复杂度和序列长度是什么关系？",
            "为什么 Transformer 用 self-attention 可以并行处理序列？",
            "self-attention 如何捕获长距离依赖？",
            "为什么需要多头 attention，而不是只用一个头？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/cross_attention": {
        "title": "Cross Attention Notes",
        "formula": [
            r"cross-attention 中 $Q$ 来自当前解码端/查询序列，$K,V$ 来自上下文序列。",
            r"$O=softmax(QK_{ctx}^\top/\sqrt{d_k})V_{ctx}$，输出长度等于 query 长度。"
        ],
        "pitfalls": [
            "误以为输出长度等于 context 长度；实际由 Q 的长度决定。",
            "Q/K/V 投影来源写混，导致 encoder-decoder 语义错误。",
            "mask 要区分 query mask、key padding mask 和 causal mask。",
            "context 长度很长时，cross-attention 也会带来较高显存成本。"
        ],
        "questions": [
            "encoder-decoder Transformer 哪些位置使用 cross-attention？",
            "cross-attention 和 self-attention 在 Q/K/V 来源上有什么区别？",
            "RAG 或多模态模型中 cross-attention 的直觉是什么？",
            "cross-attention 的输出长度由什么决定？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/multi_head_attention": {
        "title": "Multi Head Attention Notes",
        "formula": [
            r"$head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$。",
            r"$MultiHead(Q,K,V)=Concat(head_1,\ldots,head_h)W^O$。"
        ],
        "pitfalls": [
            "split/transpose 维度写错，head 维和 seq 维互换。",
            "concat 轴写错，导致输出 embedding 维不对。",
            "忘记输出投影 $W^O$。",
            "`d_model` 不能被 `num_heads` 整除时需要报错。"
        ],
        "questions": [
            "多头为什么有用？它和单头大维度 attention 有什么差异？",
            "MHA 参数量怎么估算？",
            "head_dim 为什么通常设为 `d_model / num_heads`？",
            "推理时 MHA 的主要瓶颈在哪里？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/mha_with_mask": {
        "title": "MHA With Mask Notes",
        "formula": [
            r"$scores=QK^\top/\sqrt{d_k}+mask$，其中被屏蔽位置通常为 $-\infty$。",
            r"$weights=softmax(scores)$，$O=weightsV$。"
        ],
        "pitfalls": [
            "mask 语义反了：True 是保留还是屏蔽必须在实现中固定。",
            "在 softmax 后 mask，概率和不再为 1。",
            "mask shape 没有正确广播到 batch/head/query/key 维。",
            "padding mask 和 causal mask 叠加时 dtype、shape 容易错。"
        ],
        "questions": [
            "causal mask 和 padding mask 的区别是什么？",
            "训练和自回归推理时 mask 有什么不同？",
            "为什么 mask 应该作用在 softmax 前？",
            "如果一整行都被 mask，softmax 会出现什么问题？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/sinusoidal_position_encoding": {
        "title": "Sinusoidal Position Encoding Notes",
        "formula": [
            r"$PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}})$。",
            r"$PE_{pos,2i+1}=\cos(pos/10000^{2i/d_{model}})$。"
        ],
        "pitfalls": [
            "指数写成 `i/d` 还是 `2i/d` 混乱，本实现按偶数维索引配对。",
            "奇数 `d_model` 时 cos 维度切片要对齐。",
            "位置编码要和 token embedding shape 对齐后相加。",
            "绝对位置编码外推长度时不一定和训练分布一致。"
        ],
        "questions": [
            "Transformer 为什么需要位置编码？",
            "正弦位置编码为什么使用不同频率？",
            "绝对位置编码、可学习位置编码、RoPE 有什么区别？",
            "正弦位置编码为什么被认为具备一定长度外推能力？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/rope_apply": {
        "title": "RoPE Apply Notes",
        "formula": [
            r"二维配对旋转：$[x_{2i},x_{2i+1}] \mapsto [x_{2i}\cos\theta-x_{2i+1}\sin\theta,\;x_{2i}\sin\theta+x_{2i+1}\cos\theta]$。",
            r"RoPE 通常作用于 Q/K，使 attention score 依赖相对位置。"
        ],
        "pitfalls": [
            "奇偶维配对错，导致旋转不成对。",
            "sin/cos 广播维度错，batch/head/seq 维对不上。",
            "对 V 也应用 RoPE；通常 RoPE 用于 Q/K。",
            "position index 从 0 还是从 offset 开始，在 KV cache 推理中必须一致。"
        ],
        "questions": [
            "RoPE 为什么能表达相对位置信息？",
            "RoPE 和绝对位置编码有什么区别？",
            "RoPE 在 KV cache 推理时 position offset 如何处理？",
            "RoPE 和 ALiBi 的设计直觉有什么不同？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/gqa_mqa_shapes": {
        "title": "GQA MQA Shapes Notes",
        "formula": [
            r"MHA：$n_{qheads}=n_{kvheads}$；MQA：$n_{kvheads}=1$；GQA：$1<n_{kvheads}<n_{qheads}$。",
            r"KV cache 规模近似正比于 $layers \times seq \times n_{kvheads} \times head\_dim$。"
        ],
        "pitfalls": [
            "在 seq 维 repeat，而不是在 head 维 repeat K/V。",
            "忘记 `target_heads` 必须是 `kv_heads` 的整数倍。",
            "repeat 后没有保持 head 顺序，导致 Q head 对错 KV group。",
            "只看参数量而忽略推理时 KV cache 带宽和显存。"
        ],
        "questions": [
            "MHA、MQA、GQA 的 KV cache 大小差异是什么？",
            "为什么大模型推理特别关注 KV cache？",
            "MQA 可能带来什么质量损失？GQA 如何折中？",
            "repeat_kv 和直接学习更多 KV head 的区别是什么？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/ffn_relu_gelu": {
        "title": "FFN Notes",
        "formula": [
            r"Transformer FFN 常见形式：$FFN(x)=W_2\phi(W_1x+b_1)+b_2$。",
            r"GELU 常用近似：$GELU(x)\approx0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$。"
        ],
        "pitfalls": [
            "把 FFN 和 attention 混在一起；FFN 是逐 token 的非线性变换。",
            "忘记第二个线性层，输出维度无法回到 `d_model`。",
            "GeLU 公式写错，或把输入当成 sigmoid gate。",
            "FFN expansion ratio 会显著影响参数量和计算量。"
        ],
        "questions": [
            "Transformer 中 FFN 的参数量通常占比如何？",
            "FFN 为什么可以逐 token 并行计算？",
            "GELU 相比 ReLU 的直觉优势是什么？",
            "FFN hidden size 为什么常设为 `4 * d_model` 或相近比例？"
        ],
    },
    "ml_dl_handwriting/03_attention_transformer/swiglu": {
        "title": "SwiGLU Notes",
        "formula": [
            r"$SiLU(x)=x\sigma(x)$。",
            r"$SwiGLU(x)=(SiLU(xW_g)\odot xW_u)W_d$。"
        ],
        "pitfalls": [
            "gate 和 up projection 的隐藏维度不一致。",
            "用 sigmoid 直接代替 SiLU，变成普通 GLU 变体。",
            "忘记 down projection，输出维度无法回到 `d_model`。",
            "SwiGLU 参数量通常比普通 FFN 大，hidden size 需要相应调整。"
        ],
        "questions": [
            "GLU、GeGLU、SwiGLU 的区别是什么？",
            "为什么现代 LLM 常用 gated FFN？",
            "SwiGLU 相比 ReLU/GELU FFN 增加了哪些参数？",
            "SiLU gate 的输出为什么可以看作数据依赖的通道筛选？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/causal_lm_loss": {
        "title": "Causal LM Loss Notes",
        "formula": [
            r"$L=-\frac{1}{N}\sum_t \log p_\theta(x_{t+1}|x_{\le t})$。",
            r"$PPL=\exp(CE)$，其中 CE 是按 token 平均的交叉熵。"
        ],
        "pitfalls": [
            "忘记 shift，变成预测当前 token。",
            "`ignore_index` 没处理，pad token 被计入 loss。",
            "logits/labels flatten 后顺序对不齐。",
            "loss 平均分母应只统计未被 mask 的有效 token。"
        ],
        "questions": [
            "causal LM 为什么要把 logits 和 labels shift 一位？",
            "perplexity 和 cross entropy loss 的关系是什么？",
            "训练时 pad token 为什么要 ignore？",
            "SFT 为什么通常只对 answer token 计 loss？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/sft_label_mask": {
        "title": "SFT Label Mask Notes",
        "formula": [
            r"$L=-\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}\log p_\theta(y_t|y_{<t},prompt)$。",
            r"prompt/pad 位置 label 通常设为 `ignore_index`，只让 answer token 贡献 loss。"
        ],
        "pitfalls": [
            "把 prompt 也计入 loss，模型会被训练去复述用户输入。",
            "pad 没 mask，batch padding 会污染 loss。",
            "`prompt_lengths` 和 batch 样本对不齐。",
            "多轮对话中 assistant/user/system 边界没有明确标注。"
        ],
        "questions": [
            "SFT 为什么通常只训练 answer token？",
            "多轮对话中 label mask 应如何设计？",
            "如果把 prompt token 也计入 loss，会带来什么问题？",
            "不同 chat template 会怎样影响 mask 边界？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/preference_pair_format": {
        "title": "Preference Pair Format Notes",
        "formula": [
            "偏好样本通常包含同一个 prompt 下的一对回答：`(prompt, chosen, rejected)`。",
            r"pairwise 学习关注 $\log p(chosen|prompt)-\log p(rejected|prompt)$ 的相对差异。"
        ],
        "pitfalls": [
            "`chosen`/`rejected` 顺序写反，训练信号会完全相反。",
            "prompt 没保留，后续无法拼接输入或计算条件概率。",
            "chosen/rejected 使用不同 chat template，导致比较不公平。",
            "长度差异过大时，sum logprob 和 mean logprob 的选择会影响偏好强度。"
        ],
        "questions": [
            "DPO 数据和 SFT 数据有什么区别？",
            "pairwise preference 数据如何构造和清洗？",
            "为什么 chosen/rejected 必须共享同一个 prompt？",
            "偏好数据中长度偏置会怎样影响训练？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/dpo_loss": {
        "title": "DPO Loss Notes",
        "formula": [
            r"$L_{DPO}=-\log\sigma\left(\beta[(\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x))-(\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x))]\right)$。",
            r"其中 $y_w$ 是 chosen，$y_l$ 是 rejected，$\beta$ 控制相对 reference 的约束强度。"
        ],
        "pitfalls": [
            "`chosen`/`rejected` 顺序写反，loss 会鼓励坏答案。",
            "忘记减 reference margin，退化成简单偏好分类。",
            "sigmoid/log 分开算容易数值不稳定，应使用 log-sigmoid。",
            "logprob 应只统计回答 token，prompt token 不应影响偏好差。"
        ],
        "questions": [
            "DPO 和 PPO/RLHF 的主要区别是什么？",
            "DPO 里 reference model 起什么作用？",
            "beta 变大或变小会怎样影响训练？",
            "DPO 为什么可以看作不显式训练 reward model 的偏好优化？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/reward_pairwise_loss": {
        "title": "Reward Pairwise Loss Notes",
        "formula": [
            r"Bradley-Terry 形式：$P(y_w \succ y_l)=\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))$。",
            r"pairwise loss：$L=-\log\sigma(r_w-r_l)$。"
        ],
        "pitfalls": [
            "用 MSE 拟合绝对分数，而不是学习相对偏好。",
            "`chosen`/`rejected` 顺序反了。",
            "reward 输出尺度无约束，后续 RL 或排序时需要注意校准和 clipping。",
            "只看 pair accuracy 不够，还要关注 reward hacking 和分布外泛化。"
        ],
        "questions": [
            "Reward model 为什么可以只学相对偏好？",
            "pairwise、pointwise、listwise ranking loss 有什么区别？",
            "reward 分数的绝对值有意义吗？",
            "reward model 如何被用于 PPO/RLHF？"
        ],
    },
    "ml_dl_handwriting/04_llm_training_alignment/contrastive_loss": {
        "title": "Contrastive Loss Notes",
        "formula": [
            r"InfoNCE：$L_i=-\log\frac{\exp(sim(q_i,k_i^+)/\tau)}{\sum_j\exp(sim(q_i,k_j)/\tau)}$。",
            r"若使用 cosine similarity，通常先对 embedding 做 L2 normalize。"
        ],
        "pitfalls": [
            "正负样本维度对不齐，label 对应关系错位。",
            "忘记 temperature，或 temperature 太小导致 softmax 过尖。",
            "没归一化 embedding，点积会受向量范数影响。",
            "in-batch negatives 默认其他样本都是负例，但数据中可能存在 false negative。"
        ],
        "questions": [
            "InfoNCE 和 cross entropy 的关系是什么？",
            "in-batch negatives 是什么？有什么优缺点？",
            "temperature 如何影响训练难度和梯度？",
            "为什么对比学习里常用 embedding normalize？"
        ],
    },
}


def interview_questions_body(rel: str, data: dict[str, list[str] | str]) -> str:
    questions = data["questions"]
    answers = MODULE_ANSWERS.get(rel)
    if answers is None:
        raise KeyError(f"Missing interview answers for {rel}")
    if len(answers) != len(questions):
        raise ValueError(
            f"Interview answer count mismatch for {rel}: "
            f"{len(answers)} answers for {len(questions)} questions"
        )

    blocks = []
    for question, answer in zip(questions, answers):
        blocks.append(f"::: details 参考回答：{question}\n\n{answer}\n\n:::")
    return "\n\n".join(blocks)


def notes_body(rel: str, data: dict[str, list[str] | str]) -> str:
    formula = "\n".join(f"- {item}" for item in data["formula"])
    pitfalls = "\n".join(f"- {item}" for item in data["pitfalls"])
    questions = interview_questions_body(rel, data)
    return f"""## 核心公式

{formula}

## 易错点

{pitfalls}

## 面试追问

{questions}
"""


def notes_text(rel: str, data: dict[str, list[str] | str]) -> str:
    return f"# {data['title']}\n\n{notes_body(rel, data)}"


def inline_section(rel: str, data: dict[str, list[str] | str]) -> str:
    body = notes_body(rel, data).replace("## 核心公式", "### 核心公式")
    body = body.replace("## 易错点", "### 易错点")
    body = body.replace("## 面试追问", "### 面试追问")
    return body.strip() + "\n"


def remove_source_clues(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    skip = False

    for line in lines:
        if line.startswith("## 题源线索"):
            skip = True
            continue
        if skip and line.startswith("## "):
            skip = False
            while output and output[-1] == "":
                output.pop()
            output.append("")
        if not skip:
            output.append(line)

    return "\n".join(output).strip() + "\n"


def replace_learning_notes_section(text: str, section: str) -> str:
    heading = "## 工程要点 / 面试追问"
    if heading not in text:
        return text
    before = text.split(heading, 1)[0].rstrip()
    return f"{before}\n\n{heading}\n\n{section}"


def main() -> int:
    for rel, data in MODULE_NOTES.items():
        module_dir = ROOT / rel
        name = module_dir.name
        notes_path = module_dir / "notes.md"
        page_path = module_dir / f"{name}.md"

        notes_path.write_text(notes_text(rel, data), encoding="utf-8")

        page_text = page_path.read_text(encoding="utf-8")
        page_text = remove_source_clues(page_text)
        page_text = replace_learning_notes_section(page_text, inline_section(rel, data))
        page_path.write_text(page_text, encoding="utf-8")

    print(f"Updated {len(MODULE_NOTES)} handwriting modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

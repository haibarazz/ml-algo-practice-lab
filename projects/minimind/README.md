# MiniMind 源码拆解

这个目录把 MiniMind 开源 LLM 项目拆成可阅读、可手写、可测试的学习模块。阅读顺序按 LLM 项目真实链路组织：先理解数据怎样变成 token，再理解模型怎样计算 logits 和 loss，最后理解训练、偏好对齐、推理服务与评估如何串起来。

本仓库不提交 MiniMind 源码副本；本地源码位于 `projects/minimind/source/`，该目录被 `.gitignore` 忽略。公开页面只保留源码导读、关键机制解释、手写练习和测试。

## 学习路径

1. 先读 [源码阅读指南](./SOURCE_READING_GUIDE.md)，建立完整系统地图。
2. 再读 [源码映射表](./SOURCE_MAP.md)，知道每个学习阶段对应哪些 MiniMind 文件。
3. 按 [模块索引](./MODULE_INDEX.md) 做手写练习。每个模块都对应 MiniMind 里的一个函数、类、损失、数据处理步骤或推理组件。
4. 做完练习后回到源码定位行，检查自己的实现和真实工程写法的差异。

## 分组

- [项目地图](./00_project_map/README.md)：先建立 MiniMind 的全链路视角：源码目录、训练阶段、权重流和服务入口分别承担什么职责。
- [Tokenizer 与数据](./01_tokenizer_and_data/README.md)：理解文本、对话和偏好样本怎样被整理成模型能训练的 token、label 和 mask。
- [模型结构](./02_model_architecture/README.md)：拆解 MiniMind Decoder-only Transformer 的归一化、位置编码、GQA、SwiGLU 与 MoE。
- [预训练](./03_pretraining/README.md)：理解 next-token loss、学习率调度、梯度累积、混合精度、checkpoint 和 DDP 训练骨架。
- [监督微调](./04_sft/README.md)：理解 chat template、assistant-only loss 和 thinking 标签清理怎样服务 SFT。
- [偏好对齐](./05_preference_alignment/README.md)：从 DPO 和 rollout logprob 入手，理解 chosen/rejected 偏好优化的核心张量。
- [推理与采样](./06_inference/README.md)：拆解 generate 中的 KV cache、temperature、top-k、top-p、重复惩罚和停止条件。
- [评估](./07_evaluation/README.md)：把交叉熵、困惑度和评测脚本的输出联系起来，理解语言模型基本评估量。
- [系统架构](./08_system_architecture/README.md)：把 tokenizer、数据集、模型、训练脚本、rollout 和服务脚本串成系统级依赖图。

## 阅读原则

- 先看数据流，再看模型结构；否则容易只记住层名，不知道张量从哪里来。
- 先看训练目标，再看优化技巧；否则容易把 DDP、混合精度、checkpoint 当成主线。
- 先手写最小机制，再回到完整源码；否则容易被工程细节淹没。

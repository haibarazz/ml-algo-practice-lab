#!/usr/bin/env python3
"""Refresh MiniMind source-study pages from structured notes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "minimind"


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    summary: str


@dataclass(frozen=True)
class ModuleDoc:
    key: str
    title: str
    summary: str
    sources: list[str]
    goals: list[str]
    walkthrough: list[str]
    formulas: list[str]
    practice: list[str]
    pitfalls: list[str]
    questions: list[tuple[str, str]]

    @property
    def section(self) -> str:
        return self.key.split("/", 1)[0]

    @property
    def slug(self) -> str:
        return self.key.rsplit("/", 1)[-1]


SECTIONS = [
    Section("00_project_map", "项目地图", "先建立 MiniMind 的全链路视角：源码目录、训练阶段、权重流和服务入口分别承担什么职责。"),
    Section("01_tokenizer_and_data", "Tokenizer 与数据", "理解文本、对话和偏好样本怎样被整理成模型能训练的 token、label 和 mask。"),
    Section("02_model_architecture", "模型结构", "拆解 MiniMind Decoder-only Transformer 的归一化、位置编码、GQA、SwiGLU 与 MoE。"),
    Section("03_pretraining", "预训练", "理解 next-token loss、学习率调度、梯度累积、混合精度、checkpoint 和 DDP 训练骨架。"),
    Section("04_sft", "监督微调", "理解 chat template、assistant-only loss 和 thinking 标签清理怎样服务 SFT。"),
    Section("05_preference_alignment", "偏好对齐", "从 DPO 和 rollout logprob 入手，理解 chosen/rejected 偏好优化的核心张量。"),
    Section("06_inference", "推理与采样", "拆解 generate 中的 KV cache、temperature、top-k、top-p、重复惩罚和停止条件。"),
    Section("07_evaluation", "评估", "把交叉熵、困惑度和评测脚本的输出联系起来，理解语言模型基本评估量。"),
    Section("08_system_architecture", "系统架构", "把 tokenizer、数据集、模型、训练脚本、rollout 和服务脚本串成系统级依赖图。"),
]


MODULES = [
    ModuleDoc(
        "00_project_map/project_stage_map",
        "MiniMind 项目阶段地图",
        "从源码文件名和训练脚本恢复 MiniMind 的学习顺序，先知道每一层在整条 LLM 流水线中的位置。",
        ["README.md", "README_en.md", "trainer/*.py", "dataset/lm_dataset.py", "model/model_minimind.py", "scripts/*.py"],
        [
            "按学习路径而不是文件夹字母序理解 MiniMind。",
            "区分 tokenizer、预训练、SFT、偏好对齐、RL、推理服务和评估的边界。",
            "把一个开源 LLM 项目拆成可练习的函数、数据处理步骤和训练损失。",
        ],
        [
            "MiniMind 的源码不是单一训练脚本，而是一条从数据到服务的链路。`dataset/lm_dataset.py` 负责把文本或对话样本变成 `input_ids`、`labels`、`mask`；`model/model_minimind.py` 负责把 token 变成 logits；`trainer/*.py` 决定用什么目标函数更新权重；`scripts/*.py` 负责推理、服务和格式转换。",
            "学习时应先看主干，再看分支。主干是 `PretrainDataset -> MiniMindForCausalLM -> train_pretrain.py -> generate`；分支包括 SFT 的 assistant label mask、DPO 的 chosen/rejected logprob、GRPO/PPO 的 rollout 与 reward、LoRA 的低秩增量权重。",
            "这个模块本身不是为了记文件名，而是练习“读开源项目先建地图”的能力。没有地图时，很容易把 tokenizer、模型结构、训练目标和服务接口混在一起；有地图后，每个手写题都能对应到真实源码中的一个小机制。",
        ],
        [
            "监督训练主线：`text -> tokenizer -> input_ids/labels -> model(input_ids, labels) -> loss -> optimizer.step()`。",
            "偏好优化主线：`prompt -> chosen/rejected -> policy logprob/reference logprob -> preference loss -> optimizer.step()`。",
            "推理主线：`prompt ids -> forward last token -> sample next token -> append -> update KV cache -> repeat`。",
        ],
        [
            "把源码文件归类到学习阶段，训练你在陌生项目中快速定位主线入口。",
            "每个练习模块都对应地图中的一个节点或一条边，而不是孤立算法题。",
        ],
        [
            "只按目录名读源码会漏掉跨目录依赖，例如 DPO 同时依赖 dataset、model、trainer_utils。",
            "只看 README 不足以理解训练目标，关键逻辑通常在 dataset 和 trainer 中。",
        ],
        [
            ("读一个开源 LLM 项目时，为什么要先画训练/推理阶段图？", "阶段图能先回答“数据在哪里变形、loss 在哪里计算、权重在哪里保存、推理在哪里循环”。没有这张图，读者会陷入文件细节，看懂函数却不知道它服务哪条链路。"),
            ("MiniMind 的学习顺序为什么不应该严格等于源码目录顺序？", "源码目录按工程职责组织，学习顺序要按因果链组织。比如 DPO 目录上属于 trainer，但理解它必须先懂 DPODataset、causal LM logprob 和 reference model。"),
        ],
    ),
    ModuleDoc(
        "01_tokenizer_and_data/pretrain_token_pack",
        "预训练样本打包：BOS/EOS、pad 与 label mask",
        "复刻 `PretrainDataset` 怎样把一条普通文本变成 causal LM 训练样本。",
        ["dataset/lm_dataset.py:37-55", "model/model_minimind.py:245-253"],
        [
            "理解预训练样本中 `input_ids` 和 `labels` 为什么几乎相同。",
            "理解 BOS/EOS/pad token 在固定长度 batch 中的作用。",
            "掌握 `-100` 如何让 PyTorch cross entropy 忽略 padding。",
        ],
        [
            "`PretrainDataset.__getitem__` 先取出 `sample['text']`，调用 tokenizer 得到内容 token，并为 BOS 和 EOS 预留两个位置。随后它拼接 `[BOS] + tokens + [EOS]`，再用 pad token 补齐到 `max_length`。",
            "返回的 `labels` 是 `input_ids` 的拷贝，但 pad 位置被改成 `-100`。这是因为 MiniMind 的 loss 在模型内部做 shift：logits 去掉最后一位，labels 去掉第一位，所以数据集不需要自己构造 `x/y` 两份序列。",
            "这个设计把“序列对齐”集中放在模型 loss 中，把“样本定长化”和“无效位置屏蔽”放在 dataset 中。读训练脚本时看到 `model(input_ids, labels=labels)`，就知道 shift 和 ignore 都在后续完成。",
        ],
        [
            r"打包：$ids=[BOS] + tokenizer(text)[:L-2] + [EOS] + [PAD]\times(L-|ids|)$。",
            r"标签：$labels_i=ids_i$，若 $ids_i=PAD$，则 $labels_i=-100$。",
            r"模型内 shift：用 $logits_{t}$ 预测 $labels_{t+1}$。",
        ],
        [
            "手写这个函数能把“文本样本”和“语言模型训练样本”的差异讲清楚。",
            "测试关注截断、补齐、BOS/EOS 和 pad label，这是实际训练里最常见的数据 bug。",
        ],
        [
            "忘记给 BOS/EOS 留位置，会导致末尾 EOS 被截断。",
            "labels 不屏蔽 pad，会让模型学习预测 padding。",
            "在 dataset 和 model 两边重复 shift，会产生错位两格的隐蔽错误。",
        ],
        [
            ("为什么预训练时 `input_ids` 和 `labels` 可以先设成一样？", "因为 causal LM 的错位预测在模型 loss 中完成：`logits[..., :-1, :]` 对齐 `labels[..., 1:]`。数据集只需要提供完整 token 序列和无效位置 mask。"),
            ("为什么 pad label 用 `-100` 而不是 pad token id？", "`torch.nn.functional.cross_entropy` 默认用 `ignore_index=-100` 跳过这些位置。若使用 pad token id，pad 会成为一个真实监督目标，长短样本的 padding 数量还会改变 loss 权重。"),
        ],
    ),
    ModuleDoc(
        "01_tokenizer_and_data/sft_assistant_label_mask",
        "SFT 标签掩码：只训练 assistant 回复",
        "拆解 `SFTDataset.generate_labels` 如何在多轮对话中只保留 assistant span 的监督信号。",
        ["dataset/lm_dataset.py:58-119", "trainer/train_full_sft.py:24-80"],
        [
            "理解 chat template 把 role/content/tools 转成单条训练字符串的过程。",
            "掌握 assistant 起止 token 如何决定 label mask 边界。",
            "理解为什么 SFT 不应该训练 user/system token。",
        ],
        [
            "`SFTDataset` 先通过 tokenizer 的 `apply_chat_template` 把 conversations 渲染成模型真正看到的 prompt。这个过程会插入 `<|im_start|>role`、`<|im_end|>`、tool call、think 标签等特殊结构。",
            "`generate_labels` 初始化全 `-100`，然后扫描 `bos_id = tokenizer('<|im_start|>assistant\\n')`。一旦找到 assistant 开始位置，就一直走到 `eos_id = tokenizer('<|im_end|>\\n')`，只把这段 token 复制到 labels 中。",
            "训练脚本仍然调用同一个 `model(input_ids, labels=labels)`。区别在于 labels 大部分位置都是 `-100`，因此 loss 只来自 assistant 回复。这样模型学习“在给定 system/user/tool 上下文时生成 assistant”，而不是学习复述用户输入。",
        ],
        [
            r"SFT 目标：$L=-\frac{1}{|\mathcal A|}\sum_{t\in\mathcal A}\log p_\theta(x_t|x_{<t})$。",
            r"$\mathcal A$ 是 assistant token 位置集合；非 assistant、pad、system、user 位置 label 设为 $-100$。",
        ],
        [
            "手写 mask 能训练你从模板文本中恢复监督边界。",
            "这个能力直接对应真实 SFT 数据清洗和 chat template 对齐问题。",
        ],
        [
            "assistant 起始 token 少了换行，mask 会完全扫不到。",
            "把 user/system 也计入 loss，会稀释回答部分监督，还会鼓励模型复述输入。",
            "不同 chat template 的角色边界不同，不能复用硬编码 token 序列。",
        ],
        [
            ("SFT 为什么通常只对 assistant token 计 loss？", "部署时模型的任务是在 prompt 条件下生成 assistant 回复，prompt 是条件而不是目标。只训练 assistant token 能把梯度集中到回答行为上，避免模型学习复述用户输入。"),
            ("多轮对话里 label mask 的边界最容易错在哪里？", "最容易错在 role special token 和换行。真实 tokenizer 看到的是模板渲染后的 token 序列，必须用 tokenizer 编码出来的 assistant 起止片段匹配，而不是靠字符串长度估算。"),
        ],
    ),
    ModuleDoc(
        "01_tokenizer_and_data/dpo_pair_shift",
        "DPO 偏好样本：chosen/rejected 的 x、y、mask",
        "复刻 `DPODataset` 怎样把一对偏好回答转换成 DPO 训练所需的 token 张量。",
        ["dataset/lm_dataset.py:122-192", "trainer/train_dpo.py:53-85"],
        [
            "理解 DPO 数据为什么必须成对包含 chosen 和 rejected。",
            "掌握 `x=tokens[:-1]`、`y=tokens[1:]` 和 response mask 的对齐。",
            "理解 DPO loss 只应该统计 assistant 回复 token。",
        ],
        [
            "`DPODataset` 读取同一样本中的 `chosen` 和 `rejected` 两段 conversation，分别套用同一个 chat template。这样两条序列共享 prompt 语义，差异集中在回答质量上。",
            "它对每条序列都构造三件东西：`x` 是去掉最后一个 token 的输入，`y` 是去掉第一个 token 的下一个 token 标签，`mask` 标出 assistant 回复位置。注意 mask 也要从第二个位置开始对齐 `y`。",
            "训练时 `train_dpo.py` 会把 chosen 和 rejected 在 batch 维拼接，先通过 reference model 和 policy model 分别得到每个 token 的 logprob，再按 mask 求和得到序列级 logprob。",
        ],
        [
            r"$x = ids_{0:T-1}$，$y = ids_{1:T}$，$mask = response\_mask_{1:T}$。",
            r"$\log p_\theta(y|x)=\sum_t mask_t \log p_\theta(y_t|x_{\le t})$。",
            r"chosen/rejected 必须来自同一个 prompt，否则 logprob 差混入了问题难度差异。",
        ],
        [
            "这个练习把 DPO 的数据前处理和后续序列级 loss 对齐起来。",
            "手写切片和 mask 能避免 DPO 中最常见的 off-by-one bug。",
        ],
        [
            "chosen/rejected 顺序反了，训练目标会鼓励坏回答。",
            "mask 没有跟着 shift，logprob 会统计错 token。",
            "chosen/rejected 使用不同模板，会让偏好比较不公平。",
        ],
        [
            ("DPO 数据和 SFT 数据最核心的区别是什么？", "SFT 是单个 prompt 对一个示范答案，目标是模仿；DPO 是同一 prompt 下 chosen/rejected 成对比较，目标是提高 chosen 相对 rejected 的概率优势。"),
            ("为什么 DPO 里 chosen 和 rejected 要共享同一个 prompt？", "只有共享 prompt，模型比较的才是回答质量差异。prompt 不同会把问题难度、长度和主题差异混进 logprob margin，偏好信号会变脏。"),
        ],
    ),
    ModuleDoc(
        "02_model_architecture/rms_norm_minimind",
        "RMSNorm：只按均方根缩放的归一化",
        "拆解 MiniMind 中 `RMSNorm` 的公式、数值稳定项和它在 Transformer block 中的位置。",
        ["model/model_minimind.py:50-60", "model/model_minimind.py:178-204"],
        [
            "掌握 RMSNorm 与 LayerNorm 的差异。",
            "理解 eps、float32 计算和 weight 缩放的工程意义。",
            "知道 RMSNorm 在 Pre-LN Transformer block 中怎样稳定残差流。",
        ],
        [
            "MiniMind 的 `RMSNorm.norm` 不减均值，只计算最后一维 `x.pow(2).mean(-1)`，再乘 `rsqrt(mean + eps)`。这保留了输入的均值偏移，只消除整体尺度变化。",
            "`forward` 先把输入转成 float 做归一化，再 cast 回原 dtype。这是混合精度训练中的常见写法：归一化统计对精度敏感，用 float32 更稳，而输出仍保持模型计算 dtype。",
            "在 `MiniMindBlock` 中，attention 前和 FFN 前各有一次 RMSNorm，残差连接在子层输出后相加。这是 Pre-Norm 风格，能让深层网络的梯度路径更直接。",
        ],
        [
            r"$RMS(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}$。",
            r"$RMSNorm(x)=w\odot \frac{x}{RMS(x)}$。",
            r"与 LayerNorm 相比，RMSNorm 去掉 $x-\mu$，只保留尺度归一化。",
        ],
        [
            "手写 RMSNorm 是理解现代 LLM 归一化层的入口。",
            "它比 LayerNorm 更短，但能暴露 dtype、eps、broadcast 等工程细节。",
        ],
        [
            "把 RMSNorm 写成 LayerNorm，额外减均值会改变模型行为。",
            "eps 放错位置会影响数值尺度。",
            "weight shape 应匹配 hidden size 或 head dim，广播轴不能错。",
        ],
        [
            ("为什么很多 LLM 使用 RMSNorm 而不是 BatchNorm？", "BatchNorm 依赖 batch 统计，训练和自回归推理时分布不一致；RMSNorm 对每个 token 的 hidden 维归一化，不依赖 batch，适合变长序列和小 batch。"),
            ("RMSNorm 相比 LayerNorm 少了什么不变性？", "它保留尺度不变性，但不减均值，所以不具备平移不变性。输入整体加一个常数会改变 RMSNorm 输出，而 LayerNorm 会先去中心化。"),
        ],
    ),
    ModuleDoc(
        "02_model_architecture/rope_rotate_half",
        "RoPE 旋转位置编码：MiniMind 的 rotate_half 写法",
        "从 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 理解 RoPE 怎样把位置信息注入 Q/K。",
        ["model/model_minimind.py:62-84", "model/model_minimind.py:111-124"],
        [
            "理解 MiniMind 为什么只对 Q/K 应用 RoPE。",
            "掌握 rotate_half 形式和 cos/sin 广播。",
            "理解 KV cache 推理时 position offset 为什么必须连续。",
        ],
        [
            "`precompute_freqs_cis` 为每个位置和 head 维度提前生成 cos/sin 表。MiniMind 把半维 cos/sin 复制成完整 head_dim，配合 `rotate_half` 完成二维子空间旋转。",
            "`apply_rotary_pos_emb` 的核心是 `q*cos + rotate_half(q)*sin` 和同样的 k 变换。这样 Q/K 点积会携带相对位置信息，而 V 不需要旋转，因为 V 只承载被加权汇聚的内容。",
            "在 `MiniMindModel.forward` 中，位置表切片从 `start_pos` 开始。`start_pos` 来自 past KV 长度，保证增量推理时新 token 的位置接在历史 token 后面，而不是每一步都从 0 开始。",
        ],
        [
            r"二维旋转：$(x_1,x_2)\mapsto(x_1\cos\theta-x_2\sin\theta,\ x_1\sin\theta+x_2\cos\theta)$。",
            r"MiniMind 写法：$rope(x)=x\odot cos + rotate\_half(x)\odot sin$。",
            r"RoPE 作用于 Q/K，使 attention score 依赖相对位置差。",
        ],
        [
            "手写 rotate_half 可以把 RoPE 从抽象公式落到张量维度操作。",
            "这个练习也服务后续 KV cache，因为位置 offset 是推理正确性的核心。",
        ],
        [
            "偶奇配对和前后半维配对是两种实现约定，不能混用。",
            "cos/sin 的 seq 维、head 维广播错，会导致 batch 测试不稳定。",
            "KV cache 下 position 从 0 重启，会让新旧 token 的相对位置错乱。",
        ],
        [
            ("RoPE 为什么能表达相对位置信息？", "同一旋转矩阵族作用到 Q/K 后，两者点积会出现位置角度差项。也就是说 attention score 不只取决于内容向量，还取决于 query 和 key 的相对距离。"),
            ("为什么 MiniMind 对 Q/K 旋转，而不是对 V 旋转？", "attention 权重由 Q/K 点积决定，位置信息需要影响“看谁”。V 是被权重加权求和的内容载体，通常不需要承担位置匹配。"),
        ],
    ),
    ModuleDoc(
        "02_model_architecture/repeat_kv_for_gqa",
        "GQA/MQA：重复 KV head 以服务更多 Q head",
        "拆解 MiniMind 的 `repeat_kv` 和 attention 中 `num_key_value_heads` 的显存/带宽意义。",
        ["model/model_minimind.py:86-89", "model/model_minimind.py:91-134"],
        [
            "区分 MHA、MQA、GQA 的 head 数关系。",
            "理解为什么 KV cache 大小由 KV head 数决定。",
            "掌握 `repeat_kv` 在 head 维重复而不是 seq 维重复。",
        ],
        [
            "MiniMind 配置里 `num_attention_heads` 默认是 8，`num_key_value_heads` 默认是 4，因此每两个 Q head 共享一组 K/V。这就是 GQA：Query head 多，Key/Value head 少。",
            "`repeat_kv` 接收 `[batch, seq, kv_heads, head_dim]`，先插入一个 repeat 维，再 expand 到 `n_rep`，最后 reshape 成 `[batch, seq, q_heads, head_dim]`。它是在注意力计算前把共享 K/V 展开成与 Q head 数一致的形状。",
            "推理时 KV cache 保存每层每个历史 token 的 K/V。减少 KV head 数会近似线性降低 cache 显存和读取带宽，所以 GQA 是大模型推理优化中非常关键的结构折中。",
        ],
        [
            r"MHA：$H_q=H_k=H_v$；MQA：$H_k=H_v=1$；GQA：$1 < H_{kv} < H_q$。",
            r"$n_{rep}=H_q/H_{kv}$。",
            r"KV cache 规模近似正比于 $layers \times seq \times H_{kv}\times head\_dim$。",
        ],
        [
            "手写 repeat_kv 能直接理解 GQA 的 shape 变化。",
            "这也是推理显存优化和 KV cache 章节的前置知识。",
        ],
        [
            "在 seq 维 repeat 会复制时间步，语义完全错。",
            "`num_attention_heads` 必须能被 `num_key_value_heads` 整除。",
            "只看参数量会低估 GQA 价值，它主要优化推理 cache 和带宽。",
        ],
        [
            ("GQA 相比 MHA 主要省在哪里？", "主要省 KV cache 和推理时读取 K/V 的带宽。Q head 仍然多，但 K/V head 更少，历史 token 的缓存规模随 KV head 数下降。"),
            ("`repeat_kv` 和真的学习更多 KV head 有什么区别？", "`repeat_kv` 只是把少量已学习的 K/V 表示广播给多个 Q head，不增加 K/V 参数和 cache。学习更多 KV head 会增加表达多样性，但也增加显存和带宽成本。"),
        ],
    ),
    ModuleDoc(
        "02_model_architecture/swiglu_feed_forward",
        "SwiGLU FFN：门控前馈网络",
        "拆解 MiniMind `FeedForward` 中 gate/up/down 三个投影如何组成现代 LLM 常见 FFN。",
        ["model/model_minimind.py:136-146"],
        [
            "理解普通 FFN 与 gated FFN 的差别。",
            "掌握 SwiGLU 的 gate、up、down 三段结构。",
            "理解 FFN 为什么通常是 Transformer 参数大头。",
        ],
        [
            "MiniMind 的 `FeedForward` 同时有 `gate_proj`、`up_proj` 和 `down_proj`。输入 x 分别经过 gate 和 up 两条分支，gate 分支过激活函数后与 up 分支逐元素相乘，最后再经 down 投影回 hidden size。",
            "这类结构不是简单的 `Linear -> Act -> Linear`。乘法 gate 让模型对每个 token 动态选择哪些隐藏通道通过，表达能力比纯 ReLU/GELU FFN 更强。",
            "配置中的 `intermediate_size` 默认按 `ceil(hidden_size * pi / 64) * 64` 取整，说明 MiniMind 在小模型里也保留了 LLM 常见的 FFN expansion 设计，并对硬件友好的 64 倍数做对齐。",
        ],
        [
            r"$SwiGLU(x)=W_d\left(SiLU(W_gx)\odot W_ux\right)$。",
            r"$SiLU(z)=z\sigma(z)$。",
            r"参数主要来自 $W_g,W_u,W_d$ 三个矩阵。",
        ],
        [
            "手写 SwiGLU 能理解现代 LLM FFN 为什么不是普通 MLP。",
            "练习重点是两条分支的 shape 必须一致，以及输出要回到 hidden size。",
        ],
        [
            "把 gate 分支写成普通 sigmoid GLU，会和 MiniMind 的 hidden_act 不一致。",
            "忘记 down projection 会让输出维度无法接回残差。",
            "gate/up 维度不一致时逐元素乘法会隐式广播或直接报错。",
        ],
        [
            ("SwiGLU 为什么常见于现代 LLM？", "它用数据依赖的门控控制隐藏通道，比单一激活函数更灵活。经验上在相近计算预算下，gated FFN 往往比 ReLU/GELU FFN 有更好的表达和训练效果。"),
            ("FFN 在 Transformer 中起什么作用？", "attention 负责 token 间信息交换，FFN 负责对每个 token 的表示做非线性变换。它是逐 token 独立计算的，但参数量通常占整个 block 的很大比例。"),
        ],
    ),
    ModuleDoc(
        "02_model_architecture/moe_router_aux_loss",
        "MoE 路由辅助损失：让专家负载更均衡",
        "拆解 MiniMind `MOEFeedForward` 的 top-k 路由、专家聚合和 aux loss。",
        ["model/model_minimind.py:148-176"],
        [
            "理解 MoE 中 router、expert、top-k weight 的角色。",
            "掌握 MiniMind 如何用 `index_add_` 聚合专家输出。",
            "理解 aux loss 为什么要约束专家负载均衡。",
        ],
        [
            "`MOEFeedForward` 先把 `[batch, seq, hidden]` 展平为 `[tokens, hidden]`，router 对每个 token 输出 expert 概率。`topk` 选出每个 token 要走的专家，并可把 top-k 概率重新归一化。",
            "每个 expert 只处理被分配给自己的 token。MiniMind 用 `token_idx` 找到这些 token，再把 `expert(x[token_idx]) * weight` 通过 `index_add_` 加回总输出。",
            "训练时如果没有约束，router 可能长期偏向少数 expert，导致其他 expert 学不到东西。MiniMind 的 aux loss 用实际 load 和平均 router score 的乘积鼓励专家使用更均衡。",
        ],
        [
            r"$p(e|x)=softmax(W_rx)$，选择 top-k expert。",
            r"$y_i=\sum_{e\in topk(i)} w_{i,e} Expert_e(x_i)$。",
            r"$L_{aux}=coef \cdot E \cdot \sum_e load_e \cdot score_e$。",
        ],
        [
            "手写 aux loss 能理解 MoE 不只是多个 FFN，还包括路由训练问题。",
            "这个练习对应真实源码中训练时额外加到 logits loss 上的 `aux_loss`。",
        ],
        [
            "top-k 概率不归一化时，多 expert 输出尺度可能漂移。",
            "专家没有收到 token 时仍要保持 DDP/compile 图稳定。",
            "aux loss 只在训练时有意义，推理时不应影响输出。",
        ],
        [
            ("MoE 为什么需要辅助负载均衡损失？", "router 如果只选少数专家，会造成热门专家过载、冷门专家无梯度，模型容量没有真正用起来。辅助损失给 router 一个均衡使用专家的训练信号。"),
            ("MiniMind 的 MoE 与普通 FFN 的输出维度有什么关系？", "MoE 最终仍要输出 hidden size，才能接回 Transformer 残差。区别只是中间计算由一个 FFN 变成按 token 路由到多个 expert 后再聚合。"),
        ],
    ),
    ModuleDoc(
        "03_pretraining/causal_lm_shift_loss",
        "Causal LM Shift Loss：用当前位置预测下一个 token",
        "拆解 MiniMindForCausalLM 里 logits 与 labels 的错位交叉熵。",
        ["model/model_minimind.py:245-253", "trainer/train_pretrain.py:24-80"],
        [
            "理解 next-token prediction 的 logits/labels shift。",
            "掌握 flatten 后计算 cross entropy 的形状变化。",
            "理解 ignore_index 怎样同时服务 pad 和 SFT mask。",
        ],
        [
            "`MiniMindForCausalLM.forward` 先得到 hidden states，再通过 `lm_head` 投影到 vocab logits。如果传入 labels，它会令 `x = logits[..., :-1, :]`，`y = labels[..., 1:]`，也就是第 t 个位置预测第 t+1 个 token。",
            "随后把 `[batch, seq-1, vocab]` 展平成 `[batch*(seq-1), vocab]`，把标签展平成 `[batch*(seq-1)]`，交给 `F.cross_entropy`。`ignore_index=-100` 会跳过 pad 或非 assistant 位置。",
            "预训练、SFT 和部分对齐训练都共用这个 causal LM 基础。区别不在模型 loss 公式，而在 dataset 给 labels 哪些位置填真实 token、哪些位置填 `-100`。",
        ],
        [
            r"$L=-\frac{1}{N}\sum_{t\in valid}\log p_\theta(x_{t+1}|x_{\le t})$。",
            r"$x=logits_{0:T-1}$，$y=labels_{1:T}$。",
            r"若 $labels_t=-100$，该位置不计入 $N$ 和 loss。",
        ],
        [
            "手写 shift loss 能把 LLM 最基础训练目标讲清楚。",
            "这个模块连接数据集里的 labels 和模型内部的 loss。",
        ],
        [
            "不 shift 会变成预测当前 token，模型能作弊。",
            "shift 两次会让监督错位。",
            "flatten 前后 batch/seq 顺序必须一致。",
        ],
        [
            ("为什么 causal LM 要 shift 一位？", "位置 t 的 hidden state 只能看见 t 及之前的 token，所以它的监督目标应该是下一个 token。shift 一位正是把输入上下文和下一个 token 标签对齐。"),
            ("`ignore_index=-100` 在预训练和 SFT 中分别屏蔽什么？", "预训练中主要屏蔽 padding；SFT 中还屏蔽 system/user/prompt 等非 assistant 位置。两者都通过同一个 cross entropy 参数生效。"),
        ],
    ),
    ModuleDoc(
        "03_pretraining/cosine_lr_schedule",
        "余弦学习率：从初始学习率平滑衰减",
        "复刻 `trainer_utils.get_lr` 中 MiniMind 使用的 cosine schedule。",
        ["trainer/trainer_utils.py:40-41", "trainer/train_pretrain.py:24-80"],
        [
            "理解每个 step 动态更新 optimizer learning rate。",
            "掌握 MiniMind 的余弦公式和衰减范围。",
            "理解学习率调度和训练稳定性的关系。",
        ],
        [
            "`train_epoch` 每个 step 都调用 `get_lr(epoch * iters + step, total_steps, learning_rate)`，然后把结果写回 optimizer 的每个 param group。也就是说学习率不是每个 epoch 变一次，而是按全局 step 连续变化。",
            "MiniMind 的公式是 `lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))`。当 step 为 0 时系数为 1.0；当 step 到 total_steps 时系数为 0.1，因此它从初始学习率平滑降到 10%。",
            "这个实现没有 warmup，适合小模型和简化训练脚本。读大模型训练代码时，常见变体是 warmup + cosine decay，本模块先练最核心的 cosine 部分。",
        ],
        [
            r"$lr_t=lr_0\left(0.1+0.45(1+\cos(\pi t/T))\right)$。",
            r"$t=0$ 时 $lr_t=lr_0$；$t=T$ 时 $lr_t=0.1lr_0$。",
        ],
        [
            "手写学习率函数能让你读懂训练日志里的 lr 变化。",
            "这个模块也解释为什么 train_epoch 需要知道 epoch、iters 和 step。",
        ],
        [
            "current_step 和 total_steps 如果从不同基准计数，会导致衰减过快或过慢。",
            "total_steps 为 0 时需要避免除零。",
            "只在 epoch 开头更新 lr，会和源码行为不同。",
        ],
        [
            ("为什么训练中常用余弦学习率衰减？", "它前期保持较大学习率以快速下降，后期平滑降低步长以减少震荡。相比阶梯衰减，余弦曲线没有突变，更适合长训练过程。"),
            ("MiniMind 这个公式最低为什么不是 0？", "最低保留 10% 初始学习率可以避免训练后期完全停滞。对小规模训练来说，这是一种简单保守的衰减策略。"),
        ],
    ),
    ModuleDoc(
        "03_pretraining/gradient_accumulation_counter",
        "梯度累积：用小 batch 模拟大 batch",
        "拆解 MiniMind 训练循环里 loss 缩放、反向传播和 optimizer step 的计数逻辑。",
        ["trainer/train_pretrain.py:24-80", "trainer/train_full_sft.py:24-80"],
        [
            "理解为什么 loss 要除以 accumulation_steps。",
            "掌握每隔 N 个 mini-batch 才执行 optimizer.step 的逻辑。",
            "理解 epoch 末尾剩余 batch 也需要补一次 step。",
        ],
        [
            "MiniMind 每个 mini-batch 都 forward/backward，但不是每次都更新参数。它先把 `loss = loss / accumulation_steps`，然后 `backward()` 累积梯度；只有当 `step % accumulation_steps == 0` 时才 unscale、clip、optimizer.step、zero_grad。",
            "loss 除以 accumulation_steps 是为了让累积 N 次后的梯度均值接近大 batch 一次 backward 的梯度。如果不除，等效学习率会放大 N 倍。",
            "函数末尾还有一个补偿逻辑：如果最后一个 step 不是 accumulation_steps 的整数倍，说明梯度已经累积但还没更新，需要执行一次 optimizer step。这个细节对小数据集或恢复训练很重要。",
        ],
        [
            r"等效 batch size：$B_{eff}=B_{micro}\times accumulation\_steps\times world\_size$。",
            r"每个 micro-batch 使用 $L/N$ 反传，累积 N 次后得到平均梯度。",
        ],
        [
            "这个练习把训练循环从“调库”拆成可数的状态机。",
            "它对应真实训练中显存不够但想增大有效 batch 的常见方案。",
        ],
        [
            "忘记除以 accumulation_steps，会让梯度尺度变大。",
            "忘记处理尾部剩余 batch，会丢掉最后几步梯度。",
            "zero_grad 放错位置会清空尚未累积完的梯度。",
        ],
        [
            ("梯度累积和直接增大 batch size 完全等价吗？", "在没有 BatchNorm、dropout 随机性和分布式同步差异时，梯度均值接近等价。但优化器状态更新频率、随机性和日志 step 语义仍可能不同。"),
            ("为什么梯度裁剪要放在 unscale 之后？", "混合精度下梯度可能被 scaler 放大，直接裁剪会裁到错误尺度。先 unscale 再 clip，裁剪阈值才对应真实梯度范数。"),
        ],
    ),
    ModuleDoc(
        "04_sft/empty_think_cleanup",
        "空 thinking 标签清理：控制推理风格和数据分布",
        "拆解 `post_processing_chat` 和 RLAIF/SFT 数据中 `<think>` 空块的随机保留逻辑。",
        ["dataset/lm_dataset.py:31-35", "dataset/lm_dataset.py:195-224", "trainer/train_full_sft.py:24-80"],
        [
            "理解 chat template 中 `<think>` 标签的训练语义。",
            "掌握为什么空 thinking 块不能机械全部保留或全部删除。",
            "理解数据增强式随机清理对输出格式的影响。",
        ],
        [
            "MiniMind 的 chat template 可能生成 `<think>\\n\\n</think>\\n\\n` 这种空思考块。`post_processing_chat` 以一定概率移除它，让训练数据中同时存在显式空 thinking 和直接回答两种形式。",
            "RLAIFDataset 还会通过 `open_thinking` 控制生成 prompt 是否打开 thinking。也就是说 MiniMind 不只是学回答内容，也在学“什么时候带 thinking 格式、什么时候不带”。",
            "这个模块看起来像字符串替换，但它影响的是模型输出风格和角色模板分布。对开源 LLM 项目来说，模板清理常常比模型结构更容易造成可见行为差异。",
        ],
        [
            r"若随机数 $u>empty\_think\_ratio$，将空块 `<think>\\n\\n</think>\\n\\n` 替换为空串。",
            r"保留概率约为 $empty\_think\_ratio$，删除概率约为 $1-empty\_think\_ratio$。",
        ],
        [
            "手写这个函数训练你关注 tokenizer 模板中的特殊标签。",
            "它也提示 SFT 不是只有 loss，数据格式清理本身就是训练策略。",
        ],
        [
            "机械删除所有 think 标签会改变模板分布。",
            "只做字符串替换时要保证不会误删非空推理内容。",
            "训练和推理模板不一致，会造成模型输出格式漂移。",
        ],
        [
            ("为什么空 `<think>` 标签也值得单独处理？", "它虽然没有语义内容，但会成为模型学习的输出格式。大量空 thinking 块可能让模型习惯输出空思考段，全部删除又可能失去对模板格式的适应。"),
            ("这个函数和模型能力有什么关系？", "它不改变模型结构，却改变训练分布。LLM 的行为很大一部分来自数据模板，格式清理会直接影响回答是否带 thinking、tool call 或角色标记。"),
        ],
    ),
    ModuleDoc(
        "05_preference_alignment/dpo_sequence_loss",
        "DPO 序列级损失：让 chosen 相对 rejected 更可能",
        "拆解 MiniMind `train_dpo.py` 中 reference/policy logprob margin 的计算。",
        ["trainer/train_dpo.py:25-50", "trainer/train_dpo.py:53-85"],
        [
            "掌握 DPO loss 的序列级 logprob 求和。",
            "理解 policy margin 和 reference margin 的差值。",
            "理解 beta 如何控制偏好更新强度。",
        ],
        [
            "`logits_to_log_probs` 先对 vocab 维做 log-softmax，再用 labels gather 出每个真实 token 的 logprob。`dpo_loss` 把 token logprob 乘 mask 后按序列求和，得到每条回答的条件 logprob。",
            "batch 前半是 chosen，后半是 rejected。MiniMind 先算 policy 下的 `chosen - rejected`，再算 reference 下的 `chosen - rejected`，两者相减得到 DPO logits。",
            "最终 loss 是 `-logsigmoid(beta * logits)`。当 policy 相比 reference 更偏向 chosen 时，logits 变大，loss 变小；反之则给模型梯度，推动 chosen 的相对概率上升。",
        ],
        [
            r"$\Delta_\pi=\log\pi_\theta(y_w|x)-\log\pi_\theta(y_l|x)$。",
            r"$\Delta_{ref}=\log\pi_{ref}(y_w|x)-\log\pi_{ref}(y_l|x)$。",
            r"$L=-\log\sigma(\beta(\Delta_\pi-\Delta_{ref}))$。",
        ],
        [
            "手写 DPO loss 可以把偏好优化从公式落到 batch 切片。",
            "这个模块也是理解 RLHF 替代路线的关键入口。",
        ],
        [
            "chosen/rejected batch 顺序写反会把偏好信号反过来。",
            "忘记 reference margin 会退化成简单偏好分类，约束变弱。",
            "logprob 应只统计回答 token，不能把 prompt 也算进去。",
        ],
        [
            ("DPO 里的 reference model 起什么作用？", "reference model 提供原模型的偏好基线，DPO 优化的是 policy 相对 reference 的偏好提升。它相当于隐式约束模型不要偏离原能力太远。"),
            ("beta 变大或变小有什么影响？", "beta 越大，同样的 margin 差产生更强梯度，训练更激进；beta 越小，更新更温和。过大可能过拟合偏好数据，过小可能学不动。"),
        ],
    ),
    ModuleDoc(
        "05_preference_alignment/per_token_logps",
        "逐 token logprob：rollout 后如何评估生成概率",
        "复刻 `rollout_engine.compute_per_token_logps`，理解 RL/GRPO/PPO 中生成 token 的概率记录。",
        ["trainer/rollout_engine.py:23-36", "trainer/rollout_engine.py:71-92"],
        [
            "理解为什么 rollout 后还要重新计算生成 token 的 logprob。",
            "掌握 `logits_to_keep=n_keep+1` 与 shift 的关系。",
            "理解 per-token logprob 如何服务 PPO/GRPO 的 ratio。",
        ],
        [
            "`compute_per_token_logps` 接收完整 `input_ids`，只保留最后 `n_keep` 个生成 token 的 logprob。它 forward 时传 `logits_to_keep=n_keep+1`，因为要用前一个位置的 logits 预测后一个生成 token。",
            "函数对 logits 做 log-softmax，再按真实 token id gather 出对应 logprob。返回形状是 `[batch, n_keep]`，每个元素表示模型在当时上下文下给这个生成 token 的 log 概率。",
            "在 `TorchRolloutEngine.rollout` 中，模型先 generate 出 completion，再用当前 policy 重新算 per-token logprob。后续 PPO/GRPO 会比较 old/new/ref logprob，构造 ratio、KL 或 advantage 加权目标。",
        ],
        [
            r"$\log p_\theta(y_t|x,y_{<t})=\log softmax(logits_{t-1})[y_t]$。",
            r"$ratio_t=\exp(\log p_\theta(y_t)-\log p_{old}(y_t))$。",
        ],
        [
            "这个练习是 PPO/GRPO 的前置模块：没有 per-token logprob 就没有 ratio。",
            "它也训练你处理 logits 和 token id 的 gather 维度。",
        ],
        [
            "`n_keep` 少加 1 会拿不到预测第一个生成 token 的 logits。",
            "gather 维度必须是 vocab 维。",
            "生成序列里的 pad token 需要后续 completion_mask 屏蔽。",
        ],
        [
            ("为什么 rollout 后不直接使用 generate 时的概率？", "有些推理后端不返回完整概率，或者需要用当前训练图重新计算可对齐的 logprob。重新 forward 可以确保 logprob 与当前 policy、mask 和 dtype 处理一致。"),
            ("per-token logprob 和序列 logprob 有什么关系？", "序列 logprob 通常是有效 token logprob 的和。PPO/GRPO 常保留逐 token 形式，因为 ratio、clip 和 mask 都可能按 token 粒度计算。"),
        ],
    ),
    ModuleDoc(
        "06_inference/top_k_top_p_filter",
        "Top-k / Top-p 采样过滤：控制生成候选集合",
        "拆解 MiniMind `generate` 中 temperature、top-k、top-p 和 multinomial 的采样流程。",
        ["model/model_minimind.py:257-288"],
        [
            "理解 logits 经过 temperature 后怎样改变分布尖锐程度。",
            "掌握 top-k 和 top-p 分别如何裁剪候选 token。",
            "理解采样与 greedy decoding 的差异。",
        ],
        [
            "MiniMind 每步取最后一个位置的 logits，并先除以 temperature。temperature 小于 1 会放大差异，让分布更尖；大于 1 会压平差异，让采样更随机。",
            "top-k 过滤保留 logit 最大的 k 个 token，其余置为 `-inf`。top-p 则先按概率降序累计，只保留累计概率不超过 p 的最小高概率集合，并强制保留最高概率 token。",
            "过滤后如果 `do_sample=True`，模型从 softmax 后的分布中 multinomial 采样；否则直接 argmax。这解释了为什么同一 prompt 在采样模式下可能有不同回答。",
        ],
        [
            r"temperature：$p_i=softmax(z_i/T)$。",
            r"top-k：只保留 logit 排名前 $k$ 的 token。",
            r"top-p：保留最小集合 $S$，使 $\sum_{i\in S}p_i \ge p$。",
        ],
        [
            "手写过滤逻辑可以理解生成参数对输出多样性的影响。",
            "这也是排查模型胡言乱语、重复和过保守回答的基础。",
        ],
        [
            "top-p mask 要右移，保证第一个超过阈值的 token 被保留。",
            "所有 token 都被过滤会导致 softmax NaN。",
            "temperature 不能为 0；greedy 应用 argmax 表达。",
        ],
        [
            ("top-k 和 top-p 的核心区别是什么？", "top-k 固定保留 token 数，不管分布是否尖锐；top-p 固定保留累计概率质量，候选数量会随分布形状变化。top-p 在不确定时允许更多候选，在确定时更保守。"),
            ("temperature 为什么能控制创造性？", "temperature 改变 logits 差异。低温让高概率 token 更占优势，输出稳定；高温让低概率 token 更有机会被采样，输出更多样但也更容易不稳定。"),
        ],
    ),
    ModuleDoc(
        "06_inference/kv_cache_step_slice",
        "KV cache 增量推理：每步只算新 token",
        "拆解 MiniMind generate 中 `input_ids[:, past_len:]` 和 attention 中 past K/V 拼接。",
        ["model/model_minimind.py:111-134", "model/model_minimind.py:209-232", "model/model_minimind.py:257-288"],
        [
            "理解 KV cache 为什么能降低自回归推理重复计算。",
            "掌握 `past_len` 如何决定本步输入切片。",
            "理解 K/V 拼接和 position offset 必须保持一致。",
        ],
        [
            "自回归生成第 t 步时，历史 token 的 K/V 已经在前面 step 算过。MiniMind 在 generate 中用 `past_len = past_key_values[0][0].shape[1]` 得到已缓存长度，再 forward `input_ids[:, past_len:]`，通常就是最新 token。",
            "在 Attention 里，如果有 `past_key_value`，新算出的 `xk/xv` 会和历史 K/V 在 seq 维拼接。随后再 repeat_kv、转置 head 维，进入 attention 计算。",
            "模型 forward 还用同一个 past length 作为 `start_pos`，从 RoPE cos/sin 表中切出正确位置。切片、KV 拼接和 RoPE offset 三者必须一致，否则生成会看错历史位置。",
        ],
        [
            r"无 cache 每步复杂度会重复处理整个前缀；有 cache 时每步只投影新 token 的 Q/K/V。",
            r"$K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]$。",
            r"$position_{new}$ 从 $past\_len$ 开始，而不是从 0 开始。",
        ],
        [
            "这个练习把 generate 循环和 Attention.forward 连起来。",
            "理解它后，KV cache 的 shape、显存和刷新问题会清晰很多。",
        ],
        [
            "切片错成 `input_ids[:, -1:]` 在某些多 token step 下会漏 token。",
            "position offset 从 0 重启会破坏 RoPE 相对位置。",
            "attention_mask 也要随新 token 扩展，否则 padding/可见性不一致。",
        ],
        [
            ("KV cache 为什么只缓存 K/V，不缓存 Q？", "Q 只属于当前 query 位置，每步新 token 都会产生新的 Q。历史 token 作为可被查询的记忆，需要保存的是它们的 K/V。"),
            ("KV cache 的主要代价是什么？", "代价是显存和带宽。长上下文下每层每个历史 token 都要保存 K/V，推理每步还要读取这些缓存，所以 GQA/MQA 会特别重要。"),
        ],
    ),
    ModuleDoc(
        "07_evaluation/perplexity_from_losses",
        "困惑度：从平均交叉熵到 PPL",
        "把语言模型 loss 转成更直观的 perplexity，并理解它的适用边界。",
        ["model/model_minimind.py:245-253", "eval_llm.py"],
        [
            "理解 PPL 与 token 平均 cross entropy 的关系。",
            "知道 PPL 只能在同 tokenizer、同数据分布下比较。",
            "理解 ignore token 会影响平均分母。",
        ],
        [
            "MiniMind 的训练 loss 是按有效 token 计算的 cross entropy。若这个 CE 是自然对数下的平均负 log likelihood，那么 perplexity 就是 `exp(CE)`。",
            "PPL 可以理解为模型在每个位置平均还困惑于多少个等概率候选。CE 越低，真实 token 的平均概率越高，PPL 越低。",
            "但 PPL 很依赖 tokenizer、数据集和 mask 规则。不同词表会改变 token 切分，同一段文本 token 数不同；SFT 中只算 assistant token，和预训练全 token PPL 也不能直接比较。",
        ],
        [
            r"$CE=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t|x_{<t})$。",
            r"$PPL=\exp(CE)$。",
            r"若使用以 2 为底的 log，则 $PPL=2^{CE_2}$。",
        ],
        [
            "手写 PPL 计算能把训练日志里的 loss 转成可解释指标。",
            "这个模块也提醒你不要跨 tokenizer 或跨 mask 规则比较 PPL。",
        ],
        [
            "用 batch loss 的平均再平均时，要确认每个 batch 有效 token 数是否相同。",
            "把 pad 或 prompt token 计入分母，会让指标含义变化。",
            "PPL 低不代表指令跟随、事实性或安全性一定好。",
        ],
        [
            ("PPL 为什么是 `exp(loss)`？", "因为 cross entropy 是平均负 log 概率。取指数后回到概率空间，可以解释为模型平均每步的有效候选数。"),
            ("什么时候 PPL 不适合比较两个模型？", "当 tokenizer、评测文本、mask 规则或上下文长度不同的时候，PPL 不再是同一分布下的同一指标。特别是聊天 SFT 和预训练 PPL 不能简单横向比较。"),
        ],
    ),
    ModuleDoc(
        "08_system_architecture/training_pipeline_edges",
        "MiniMind 系统依赖图：数据流、权重流和服务流",
        "把 MiniMind 的源码节点串成可执行的系统架构图，理解训练产物如何进入推理服务。",
        ["dataset/lm_dataset.py", "model/model_minimind.py", "trainer/*.py", "scripts/*.py", "eval_llm.py"],
        [
            "理解数据、模型、训练脚本、checkpoint、推理服务之间的依赖。",
            "区分训练流、对齐流、rollout 流和服务流。",
            "掌握从源码中抽取系统级边的能力。",
        ],
        [
            "训练流从 dataset 开始：不同 Dataset 产出不同 batch 结构，预训练/SFT 是 `(input_ids, labels)`，DPO 是 chosen/rejected 的 x/y/mask，RL/Agent 则先产出 prompt/messages/tools，再 rollout。",
            "模型流集中在 `MiniMindForCausalLM`：训练时返回 loss/logits/aux_loss，推理时通过 generate 循环返回 generated_ids 和可选 past_kv。MoE 模型还会在训练目标中额外加入 aux_loss。",
            "权重流由 `trainer_utils.lm_checkpoint` 和各训练脚本共同管理：训练脚本保存 `.pth`，可从 `from_weight` 加载已有权重继续 SFT/DPO/LoRA。服务流再通过 `eval_llm.py`、`serve_openai_api.py`、`web_demo.py` 读取模型并暴露交互入口。",
        ],
        [
            "预训练边：`PretrainDataset -> MiniMindForCausalLM(loss) -> AdamW -> checkpoint`。",
            "SFT 边：`SFTDataset(label mask) -> same causal LM loss -> full_sft weight`。",
            "DPO 边：`DPODataset -> policy/ref logprob -> DPO loss -> dpo weight`。",
            "推理边：`tokenizer prompt -> generate -> KV cache -> sampled tokens -> decode`。",
        ],
        [
            "这个练习训练你从工程代码中抽取系统图，而不是只读单个函数。",
            "系统图能帮助读者判断新源码机制应该挂在哪条主线下。",
        ],
        [
            "把所有 trainer 都看成同一个训练脚本，会忽略 batch 结构和 loss 差异。",
            "只画数据流不画权重流，会看不懂 `from_weight` 和阶段继承。",
            "只画训练不画服务，会漏掉 tokenizer/template 在推理中的一致性问题。",
        ],
        [
            ("MiniMind 的系统图里为什么要区分数据流和权重流？", "数据流解释一个 batch 如何进入 loss，权重流解释不同训练阶段如何继承和保存模型。SFT、DPO、LoRA 的关键差别很多时候不在 forward，而在权重从哪里来、保存到哪里去。"),
            ("为什么服务流也应该放进学习地图？", "LLM 项目的目标不是只训练出 loss，而是能推理、对话或提供 API。服务流会暴露 tokenizer、chat template、KV cache、采样参数等与训练同样关键的工程约束。"),
        ],
    ),
]


SECTION_BY_KEY = {section.key: section for section in SECTIONS}
MODULE_BY_KEY = {module.key: module for module in MODULES}


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def numbered(items: list[str]) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, 1))


def detail_blocks(questions: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"::: details 参考回答：{question}\n\n{answer}\n\n:::"
        for question, answer in questions
    )


def normalize_exercise_block(text: str) -> str:
    start_marker = "## 带提示练习区"
    end_marker = "## 工程要点 / 面试追问"
    if start_marker not in text:
        raise ValueError("missing exercise block")

    block = text.split(start_marker, 1)[1]
    if end_marker in block:
        block = block.split(end_marker, 1)[0]
    block = start_marker + block
    replacements = {
        "> Status: complete\n\n": "",
        "## STOP HERE": "## 先停在这里",
        "TODO guided implementation.": "带提示实现。",
        "TODO blank implementation.": "无提示实现。",
        "All tests passed.": "All tests passed.",
    }
    for old, new in replacements.items():
        block = block.replace(old, new)
    return block.strip()


def notes_body(module: ModuleDoc) -> str:
    return f"""## 关键公式与数据流

{bullets(module.formulas)}

## 易错点

{bullets(module.pitfalls)}

## 面试追问

{detail_blocks(module.questions)}
"""


def module_page(module: ModuleDoc, exercise_block: str) -> str:
    return f"""# {module.title}

{module.summary}

## 学习目标

{bullets(module.goals)}

## MiniMind 源码定位

{bullets(f'`{source}`' for source in module.sources)}

## 源码机制详解

{chr(10).join(module.walkthrough)}

## 关键公式与数据流

{bullets(module.formulas)}

## 为什么练这个

{bullets(module.practice)}

{exercise_block}

## 工程要点 / 面试追问

{notes_body(module).replace('## ', '### ').strip()}
"""


def root_readme() -> str:
    section_lines = "\n".join(
        f"- [{section.title}](./{section.key}/README.md)：{section.summary}"
        for section in SECTIONS
    )
    return f"""# MiniMind 源码拆解

这个目录把 MiniMind 开源 LLM 项目拆成可阅读、可手写、可测试的学习模块。阅读顺序按 LLM 项目真实链路组织：先理解数据怎样变成 token，再理解模型怎样计算 logits 和 loss，最后理解训练、偏好对齐、推理服务与评估如何串起来。

本仓库不提交 MiniMind 源码副本；本地源码位于 `projects/minimind/source/`，该目录被 `.gitignore` 忽略。公开页面只保留源码导读、关键机制解释、手写练习和测试。

## 学习路径

1. 先读 [源码阅读指南](./SOURCE_READING_GUIDE.md)，建立完整系统地图。
2. 再读 [源码映射表](./SOURCE_MAP.md)，知道每个学习阶段对应哪些 MiniMind 文件。
3. 按 [模块索引](./MODULE_INDEX.md) 做手写练习。每个模块都对应 MiniMind 里的一个函数、类、损失、数据处理步骤或推理组件。
4. 做完练习后回到源码定位行，检查自己的实现和真实工程写法的差异。

## 分组

{section_lines}

## 阅读原则

- 先看数据流，再看模型结构；否则容易只记住层名，不知道张量从哪里来。
- 先看训练目标，再看优化技巧；否则容易把 DDP、混合精度、checkpoint 当成主线。
- 先手写最小机制，再回到完整源码；否则容易被工程细节淹没。
"""


def source_reading_guide() -> str:
    return """# MiniMind 源码阅读指南

MiniMind 的学习价值不只是“有一个小 LLM 模型文件”，而是它把一个 LLM 项目从 tokenizer、数据集、模型结构、训练脚本、偏好优化、推理服务到评估都放在同一个较小代码库里。读它时要按链路读，而不是按目录逐个打开。

## 1. 数据先进入 tokenizer 和 Dataset

文本不会直接进入模型。`trainer/train_tokenizer.py` 展示 BPE tokenizer 的训练方式，`dataset/lm_dataset.py` 决定不同阶段的样本如何变成张量：

- 预训练样本：`text -> [BOS] + tokens + [EOS] + [PAD] -> labels`。
- SFT 样本：`conversations -> chat template -> input_ids -> assistant-only labels`。
- DPO 样本：`chosen/rejected -> x/y shift -> response mask`。
- RLAIF/Agent 样本：`prompt/messages/tools -> rollout -> reward 或规则判断`。

读这一层时要关注三个问题：哪些 token 参与 loss，哪些 token 只是条件，哪些位置必须被 mask。

## 2. 模型主干集中在 `model/model_minimind.py`

MiniMind 是 Decoder-only Causal LM。主干顺序是：

```text
input_ids
-> token embedding
-> N 个 MiniMindBlock
   -> RMSNorm
   -> GQA attention + RoPE + KV cache
   -> RMSNorm
   -> SwiGLU FFN 或 MoE FFN
-> final RMSNorm
-> LM Head
-> logits / loss / generate
```

这一层要重点读：

- `MiniMindConfig`：模型宽度、层数、head 数、KV head 数、RoPE、MoE 参数。
- `RMSNorm`：LLM 常用归一化，不依赖 batch。
- `precompute_freqs_cis` / `apply_rotary_pos_emb`：RoPE 位置编码。
- `repeat_kv` / `Attention`：GQA/MQA 和 KV cache 的核心 shape。
- `FeedForward` / `MOEFeedForward`：SwiGLU 与 MoE 路由。
- `MiniMindForCausalLM.forward`：next-token loss 的 shift。
- `generate`：temperature、top-k、top-p、repetition penalty、KV cache。

## 3. 训练脚本是一套共享骨架加不同数据/目标

`trainer/train_pretrain.py`、`trainer/train_full_sft.py`、`trainer/train_lora.py` 的外层循环非常相似：初始化分布式环境、构造模型和数据、设置混合精度、计算 loss、梯度累积、保存 checkpoint。差异主要来自 Dataset 和参数冻结方式。

核心训练公式是：

```text
loss = task_loss + aux_loss
loss = loss / accumulation_steps
backward
每 accumulation_steps 次执行 optimizer.step
```

MoE 打开时，`aux_loss` 来自专家路由均衡；不开 MoE 时它是 0。

## 4. 偏好优化从 token logprob 开始

DPO、PPO、GRPO 不是从“生成一句话好不好”直接更新模型，而是先把生成序列拆回 token 级 logprob：

```text
logits -> log_softmax -> gather(token_id) -> per-token logprob
```

DPO 比较 chosen/rejected 的序列 logprob margin；PPO/GRPO 比较新旧策略 logprob ratio，并结合 reward 或 advantage。`trainer/rollout_engine.py` 是连接 generate 和 RL loss 的关键文件。

## 5. 推理服务必须和训练模板一致

`eval_llm.py`、`scripts/web_demo.py`、`scripts/serve_openai_api.py` 都要依赖同一个 tokenizer 和 chat template。训练时 assistant span 怎么标记，推理时 prompt 就必须怎么组织；训练时使用哪些 special token，服务端也要一致。

推理性能的核心不是只看模型参数量，还要看：

- KV cache 的长度和 head 数。
- top-k/top-p/temperature 的采样策略。
- 是否使用流式输出。
- 是否把权重转换为 transformers 格式或服务端格式。

## 建议阅读顺序

1. `dataset/lm_dataset.py`
2. `model/model_minimind.py`
3. `trainer/trainer_utils.py`
4. `trainer/train_pretrain.py`
5. `trainer/train_full_sft.py`
6. `trainer/train_dpo.py`
7. `trainer/rollout_engine.py`
8. `model/model_lora.py`
9. `scripts/serve_openai_api.py` / `scripts/web_demo.py`
10. `eval_llm.py` / `scripts/eval_toolcall.py`
"""


def source_map() -> str:
    rows = [
        ("项目地图", "`README.md`, `README_en.md`, `trainer/*.py`", "训练阶段、权重继承、推理入口和发布格式"),
        ("Tokenizer 与数据", "`trainer/train_tokenizer.py`, `dataset/lm_dataset.py`", "BPE、chat template、Pretrain/SFT/DPO/RLAIF/Agent 数据格式"),
        ("模型结构", "`model/model_minimind.py`, `model/model_lora.py`", "RMSNorm、RoPE、GQA、SwiGLU、MoE、LoRA、LM Head"),
        ("预训练", "`trainer/train_pretrain.py`, `trainer/trainer_utils.py`", "Causal LM loss、学习率、梯度累积、混合精度、DDP、checkpoint"),
        ("监督微调", "`trainer/train_full_sft.py`, `trainer/train_lora.py`, `dataset/lm_dataset.py`", "assistant-only labels、全参 SFT、LoRA 参数训练"),
        ("偏好对齐", "`trainer/train_dpo.py`, `trainer/rollout_engine.py`", "chosen/rejected、DPO margin、per-token logprob"),
        ("RL 与 Agent", "`trainer/train_ppo.py`, `trainer/train_grpo.py`, `trainer/train_agent.py`", "rollout、reward、advantage、tool call loop"),
        ("推理服务", "`model/model_minimind.py`, `scripts/serve_openai_api.py`, `scripts/web_demo.py`, `scripts/chat_api.py`", "KV cache、采样过滤、OpenAI API、Web UI"),
        ("评估", "`eval_llm.py`, `scripts/eval_toolcall.py`", "PPL、benchmark 调用、tool call 用例评估"),
    ]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return f"""# MiniMind 源码映射表

本表按学习路径映射 MiniMind 源码。`projects/minimind/source/` 是本地源码副本，不会提交到仓库；公开文档只引用文件路径和关键行号。

| 学习阶段 | 主要源码 | 重点问题 |
| --- | --- | --- |
{table}

## 模块覆盖范围

模块覆盖数据打包、SFT mask、DPO 数据、RMSNorm、RoPE、GQA、SwiGLU、MoE、causal LM loss、学习率、梯度累积、thinking 标签、DPO loss、per-token logprob、采样过滤、KV cache、PPL 和系统依赖图。

扩展阅读源码包括 `model/model_lora.py`、`trainer/train_lora.py`、`trainer/train_ppo.py`、`trainer/train_grpo.py`、`trainer/train_agent.py`、`scripts/serve_openai_api.py` 和 `scripts/convert_model.py`。
"""


def module_index() -> str:
    lines = ["# MiniMind 模块索引", "", "按源码学习路径排列。每个模块都包含源码导读、关键公式、手写练习、测试和参考答案。", ""]
    for section in SECTIONS:
        lines.extend([f"## {section.title}", "", section.summary, ""])
        for module in MODULES:
            if module.section == section.key:
                lines.append(f"- [{module.title}](./{module.key}/{module.slug}.md)：{module.summary}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def section_readme(section: Section) -> str:
    rows = []
    for module in MODULES:
        if module.section == section.key:
            rows.append(f"| [{module.title}](./{module.slug}/{module.slug}.md) | {module.summary} |")
    return f"""# {section.title}

{section.summary}

| 模块 | 学习重点 |
| --- | --- |
{chr(10).join(rows)}
"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    write_text(PROJECT / "README.md", root_readme())
    write_text(PROJECT / "SOURCE_READING_GUIDE.md", source_reading_guide())
    write_text(PROJECT / "SOURCE_MAP.md", source_map())
    write_text(PROJECT / "MODULE_INDEX.md", module_index())

    for section in SECTIONS:
        write_text(PROJECT / section.key / "README.md", section_readme(section))

    for module in MODULES:
        module_dir = PROJECT / module.key
        page_path = module_dir / f"{module.slug}.md"
        original = page_path.read_text(encoding="utf-8")
        exercise = normalize_exercise_block(original)
        write_text(page_path, module_page(module, exercise))
        write_text(module_dir / "notes.md", f"# {module.title}笔记\n\n{notes_body(module)}")

    print(f"Refreshed MiniMind docs: {len(SECTIONS)} sections, {len(MODULES)} modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# MiniMind 项目阶段地图

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/00_project_map/project_stage_map/project_stage_map.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


从源码文件名和训练脚本恢复 MiniMind 的学习顺序，先知道每一层在整条 LLM 流水线中的位置。

## 学习目标

- 按学习路径而不是文件夹字母序理解 MiniMind。
- 区分 tokenizer、预训练、SFT、偏好对齐、RL、推理服务和评估的边界。
- 把一个开源 LLM 项目拆成可练习的函数、数据处理步骤和训练损失。

## MiniMind 源码定位

- `README.md`
- `README_en.md`
- `trainer/*.py`
- `dataset/lm_dataset.py`
- `model/model_minimind.py`
- `scripts/*.py`

## 源码机制详解

MiniMind 的源码不是单一训练脚本，而是一条从数据到服务的链路。`dataset/lm_dataset.py` 负责把文本或对话样本变成 `input_ids`、`labels`、`mask`；`model/model_minimind.py` 负责把 token 变成 logits；`trainer/*.py` 决定用什么目标函数更新权重；`scripts/*.py` 负责推理、服务和格式转换。
学习时应先看主干，再看分支。主干是 `PretrainDataset -> MiniMindForCausalLM -> train_pretrain.py -> generate`；分支包括 SFT 的 assistant label mask、DPO 的 chosen/rejected logprob、GRPO/PPO 的 rollout 与 reward、LoRA 的低秩增量权重。
这个模块本身不是为了记文件名，而是练习“读开源项目先建地图”的能力。没有地图时，很容易把 tokenizer、模型结构、训练目标和服务接口混在一起；有地图后，每个手写题都能对应到真实源码中的一个小机制。

## 关键公式与数据流

- 监督训练主线：`text -> tokenizer -> input_ids/labels -> model(input_ids, labels) -> loss -> optimizer.step()`。
- 偏好优化主线：`prompt -> chosen/rejected -> policy logprob/reference logprob -> preference loss -> optimizer.step()`。
- 推理主线：`prompt ids -> forward last token -> sample next token -> append -> update KV cache -> repeat`。

## 为什么练这个

- 把源码文件归类到学习阶段，训练你在陌生项目中快速定位主线入口。
- 每个练习模块都对应地图中的一个节点或一条边，而不是孤立算法题。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_project_stage_map(files):
    """带提示实现。"""
    # TODO 1: 定义阶段顺序
    # TODO 2: 用文件路径关键词归类
    # TODO 3: 忽略无关文件并保持稳定顺序
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_project_stage_map(files):
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
def test_build_project_stage_map():
    files = [
        "model/model_minimind.py",
        "trainer/train_dpo.py",
        "dataset/lm_dataset.py",
        "scripts/serve_openai_api.py",
        "trainer/train_pretrain.py",
        "README.md",
    ]
    stages = build_project_stage_map(files)
    assert stages == ["project_map", "data", "model", "pretrain", "preference_alignment", "inference"]


test_build_project_stage_map()
print("All tests passed.")
```

## 先停在这里

请先完成带提示练习区和无提示练习区，再查看参考答案。

::: details 点击查看参考答案与解析

## 参考答案与解析

```python
def build_project_stage_map(files):
    order = [
        ("project_map", ("README", "dataset.md")),
        ("data", ("dataset/", "train_tokenizer")),
        ("model", ("model/", "model_minimind")),
        ("pretrain", ("train_pretrain",)),
        ("sft", ("train_full_sft", "train_lora")),
        ("preference_alignment", ("train_dpo", "train_ppo", "train_grpo", "train_agent", "rollout_engine")),
        ("inference", ("serve_openai_api", "web_demo", "chat_api")),
        ("evaluation", ("eval_llm", "eval_toolcall")),
    ]
    seen = set()
    for stage, keys in order:
        for f in files:
            if any(k in f for k in keys):
                seen.add(stage)
                break
    return [stage for stage, _ in order if stage in seen]
```

### 解析

1. 面试追问：为什么不按目录顺序学？
2. 系统设计追问：source map 为什么要保留？


:::

## 工程要点 / 面试追问

### 关键公式与数据流

- 监督训练主线：`text -> tokenizer -> input_ids/labels -> model(input_ids, labels) -> loss -> optimizer.step()`。
- 偏好优化主线：`prompt -> chosen/rejected -> policy logprob/reference logprob -> preference loss -> optimizer.step()`。
- 推理主线：`prompt ids -> forward last token -> sample next token -> append -> update KV cache -> repeat`。

### 易错点

- 只按目录名读源码会漏掉跨目录依赖，例如 DPO 同时依赖 dataset、model、trainer_utils。
- 只看 README 不足以理解训练目标，关键逻辑通常在 dataset 和 trainer 中。

### 面试追问

::: details 参考回答：读一个开源 LLM 项目时，为什么要先画训练/推理阶段图？

阶段图能先回答“数据在哪里变形、loss 在哪里计算、权重在哪里保存、推理在哪里循环”。没有这张图，读者会陷入文件细节，看懂函数却不知道它服务哪条链路。

:::

::: details 参考回答：MiniMind 的学习顺序为什么不应该严格等于源码目录顺序？

源码目录按工程职责组织，学习顺序要按因果链组织。比如 DPO 目录上属于 trainer，但理解它必须先懂 DPODataset、causal LM logprob 和 reference model。

:::
# Project Stage Map

::: tip 云端运行环境

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haibarazz/ml-algo-practice-lab/blob/main/projects/minimind/00_project_map/project_stage_map/project_stage_map.ipynb)
[![Open In Studio](https://img.shields.io/badge/Open%20In-ModelScope-blueviolet?logo=alibabacloud)](https://modelscope.cn/my/mynotebook)

:::


> Status: complete

## Source Mapping

- `README.md:82-94`
- `model/model_minimind.py:10-288`
- `dataset/lm_dataset.py:37-252`
- `trainer/train_pretrain.py:83-171`
- `trainer/train_full_sft.py:84-172`
- `trainer/train_dpo.py:25-50`
- `scripts/serve_openai_api.py:28-245`

## 手写实现约束

允许使用 Python list/dict/string；不需要读取真实文件系统。

## 原理最小说明

MiniMind 可以先按训练链路理解：tokenizer/data -> model -> pretrain -> SFT -> preference/RL -> inference -> evaluation/system。这个模块练的是把散落文件映射到学习阶段，而不是背目录。

## 带提示练习区

先按 TODO 补全下面的函数。这个版本保留实现台阶。

```python
def build_project_stage_map(files):
    """TODO guided implementation."""
    # TODO 1: 定义阶段顺序
    # TODO 2: 用文件路径关键词归类
    # TODO 3: 忽略无关文件并保持稳定顺序
    raise NotImplementedError
```

## 无提示练习区

撤掉提示后，独立实现同一个功能。

```python
def build_project_stage_map(files):
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

## STOP HERE

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

见 `notes.md`。
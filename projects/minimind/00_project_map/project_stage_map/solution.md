# Project Stage Map Solution

## Reference Implementation

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

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。

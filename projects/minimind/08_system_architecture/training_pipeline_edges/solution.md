# Training Pipeline Edges Solution

## Reference Implementation

```python
def build_training_pipeline_edges():
    return [
        ("raw_text", "tokenizer"),
        ("tokenizer", "pretrain_dataset"),
        ("tokenizer", "sft_dataset"),
        ("model_config", "model_architecture"),
        ("pretrain_dataset", "pretrain_training"),
        ("model_architecture", "pretrain_training"),
        ("pretrain_training", "pretrained_weights"),
        ("pretrained_weights", "sft_training"),
        ("sft_dataset", "sft_training"),
        ("sft_training", "sft_weights"),
        ("sft_weights", "dpo_training"),
        ("preference_pairs", "dpo_training"),
        ("dpo_training", "aligned_weights"),
        ("aligned_weights", "openai_api_server"),
        ("aligned_weights", "evaluation"),
    ]
```

## Explanation

这份答案是 MiniMind 源码机制的最小可手写版本，和真实项目的差异主要在于：这里用小输入和轻量依赖来隔离核心算法，不负责完整训练工程。

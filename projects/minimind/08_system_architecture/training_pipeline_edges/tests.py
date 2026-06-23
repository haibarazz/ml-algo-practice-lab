"""Lightweight tests for training_pipeline_edges.

Run with: python tests.py
"""

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


def test_build_training_pipeline_edges():
    edges = build_training_pipeline_edges()
    assert ("raw_text", "tokenizer") in edges
    assert ("tokenizer", "pretrain_dataset") in edges
    assert ("pretrained_weights", "sft_training") in edges
    assert ("aligned_weights", "openai_api_server") in edges


test_build_training_pipeline_edges()
print("All tests passed.")

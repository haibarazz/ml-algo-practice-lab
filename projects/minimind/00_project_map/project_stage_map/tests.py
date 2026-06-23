"""Lightweight tests for project_stage_map.

Run with: python tests.py
"""

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

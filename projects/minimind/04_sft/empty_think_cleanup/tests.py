"""Lightweight tests for empty_think_cleanup.

Run with: python tests.py
"""

def cleanup_empty_think(prompt, keep_empty=False):
    pattern = "<think>\\n\\n</think>\\n\\n"
    if keep_empty:
        return prompt
    return prompt.replace(pattern, "")


def test_cleanup_empty_think():
    text = "A<think>\\n\\n</think>\\n\\nB"
    assert cleanup_empty_think(text, keep_empty=True) == text
    assert cleanup_empty_think(text, keep_empty=False) == "AB"


test_cleanup_empty_think()
print("All tests passed.")

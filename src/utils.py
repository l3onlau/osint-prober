"""
Shared utility functions used across the project.
"""

import re


def strip_think_tags(text: str) -> str:
    """Removes Qwen3 <think>…</think> reasoning blocks from LLM output.

    Qwen3 models emit chain-of-thought wrapped in these tags before the
    final answer. This must be stripped before JSON extraction or display.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

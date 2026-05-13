from __future__ import annotations

from src.utils.llm_client import build_llm_prompt


def test_build_llm_prompt_includes_label_and_notes() -> None:
    prompt = build_llm_prompt("Normal Gait", "All metrics stable", user_input="Hello")
    assert "Normal Gait" in prompt
    assert "All metrics stable" in prompt
    assert "Hello" in prompt

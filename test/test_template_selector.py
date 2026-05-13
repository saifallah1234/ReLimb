from __future__ import annotations

from src.utils.template_selector import pick_template


def test_pick_template_is_deterministic() -> None:
    templates = {"Normal Gait": ["A", "B", "C"]}
    idx1, template1 = pick_template("Normal Gait", "video-123", templates)
    idx2, template2 = pick_template("Normal Gait", "video-123", templates)

    assert idx1 == idx2
    assert template1 == template2

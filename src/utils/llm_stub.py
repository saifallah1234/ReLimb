from __future__ import annotations


def generate_summary_from_template(label: str, template: str) -> str:
    return template.replace("{label}", label)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "data" / "templates.json"


def _hash_to_index(key: str, n: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n


def load_templates(path: Path | None = None) -> dict[str, list[str]]:
    if path is None:
        path = DEFAULT_TEMPLATE_PATH
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return {k: list(v) for k, v in data.items()}


def pick_template(label: str, video_id: str, templates: dict[str, list[str]]) -> tuple[int, str]:
    options = templates.get(label) or templates.get("Unknown / Other") or []
    if not options:
        raise ValueError("No templates available for label selection.")
    idx = _hash_to_index(f"{label}:{video_id}", len(options))
    return idx, options[idx]

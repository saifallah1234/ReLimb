from __future__ import annotations

from typing import Iterable


def predict_label_stub(class_labels: Iterable[str] | None = None, default_label: str = "Unknown / Other") -> str:
    if class_labels is None:
        return default_label
    labels = list(class_labels)
    return labels[0] if labels else default_label

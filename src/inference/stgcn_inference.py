from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.data.datasets.relimb_dataset import clean_keypoints, filter_gait_keypoints
from src.models.model_stgcn import GaitSTGCN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "stgcn_best.pth"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "data" / "class_mapping.json"


def _load_class_mapping(mapping_path: Path | None = None) -> dict[int, str]:
    if mapping_path is None:
        mapping_path = DEFAULT_MAPPING_PATH
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(v): k for k, v in mapping.items()}


def _prepare_keypoints(keypoints_normalized: np.ndarray) -> np.ndarray:
    cleaned = clean_keypoints(keypoints_normalized)
    filtered = filter_gait_keypoints(cleaned)
    return filtered.astype(np.float32)


def _load_checkpoint(model_path: Path) -> tuple[dict, int]:
    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(model_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        num_classes = int(checkpoint.get("num_classes", 0) or 0)
        return checkpoint, num_classes
    if isinstance(checkpoint, dict):
        return {"model": checkpoint}, 0
    raise ValueError("Unsupported checkpoint format.")


def _resolve_label_names(num_classes: int, label_names: Iterable[str] | None = None) -> list[str]:
    if label_names is not None:
        names = list(label_names)
        if len(names) != num_classes:
            raise ValueError("label_names length does not match num_classes.")
        return names

    mapping = _load_class_mapping()
    ordered = [mapping[i] for i in sorted(mapping.keys())]
    if num_classes and num_classes != len(ordered):
        return ordered[:num_classes]
    return ordered


def predict_label(
    keypoints_normalized: np.ndarray,
    model_path: Path | None = None,
    label_names: Iterable[str] | None = None,
) -> tuple[str, float]:
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    checkpoint, num_classes = _load_checkpoint(model_path)
    if num_classes == 0:
        num_classes = len(_load_class_mapping())

    model = GaitSTGCN(num_joints=13, num_classes=num_classes)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    input_np = _prepare_keypoints(keypoints_normalized)
    input_tensor = torch.from_numpy(input_np).unsqueeze(0)  # (1, T, 26)
    lengths = torch.tensor([input_np.shape[0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_tensor, lengths=lengths)
        probs = torch.softmax(logits, dim=-1)
        score, idx = torch.max(probs, dim=-1)

    labels = _resolve_label_names(num_classes, label_names)
    label = labels[int(idx.item())] if labels else "Unknown / Other"
    return label, float(score.item())
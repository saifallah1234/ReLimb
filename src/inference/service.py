from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pose_extraction.single_video_pipeline import run_pipeline


def run_video_inference(
    video_path: Path,
    video_id: str | None = None,
    output_dir: Path | None = None,
    fixed_label: str | None = None,
    user_input: str = "",
) -> dict[str, Any]:
    normalized_path = run_pipeline(
        video_path=video_path,
        video_id=video_id,
        output_dir=output_dir,
        fixed_label=fixed_label,
        user_input=user_input,
    )

    session_folder = normalized_path.parent
    prediction_path = session_folder / "prediction.json"
    if not prediction_path.exists():
        raise FileNotFoundError("Prediction output not found after inference.")

    with prediction_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "video_id": payload.get("video_id", session_folder.name),
        "label": payload.get("label", "Unknown / Other"),
        "summary": payload.get("summary", ""),
        "score": payload.get("score"),
        "prediction_path": str(prediction_path),
    }

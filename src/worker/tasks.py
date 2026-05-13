from __future__ import annotations

from pathlib import Path
from typing import Any

from celery import shared_task

from src.core.settings import settings
from src.inference.service import run_video_inference


@shared_task(name="relimgait.run_prediction")
def run_prediction(
    video_path: str,
    video_id: str | None = None,
    user_input: str = "",
) -> dict[str, Any]:
    return run_video_inference(
        video_path=Path(video_path),
        video_id=video_id,
        output_dir=settings.results_dir,
        fixed_label=settings.fixed_label,
        user_input=user_input,
    )

from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    video_id: str
    label: str
    summary: str
    score: float | None = None
    prediction_path: str | None = None

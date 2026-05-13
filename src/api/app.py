from __future__ import annotations

from pathlib import Path
from typing import Any

from celery.exceptions import TimeoutError as CeleryTimeoutError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.schemas import PredictionResponse
from src.api.storage import save_upload
from src.core.settings import settings
from src.worker.celery_app import celery_app

app = FastAPI(title="ReLimb Inference API")


@app.post("/predict", response_model=PredictionResponse)
def predict(
    video: UploadFile = File(...),
    user_input: str = Form(""),
) -> PredictionResponse:
    try:
        saved_path, video_id = save_upload(video, settings.incoming_dir)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    task = celery_app.send_task(
        "relimgait.run_prediction",
        args=[str(saved_path), video_id, user_input],
    )

    try:
        result: dict[str, Any] = task.get(timeout=settings.task_timeout_s)
    except CeleryTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Prediction timed out.") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(**result)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

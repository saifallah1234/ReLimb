from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("RELIMB_REDIS_URL", "redis://localhost:6379/0")
    celery_backend_url: str = os.getenv("RELIMB_CELERY_BACKEND", redis_url)
    incoming_dir: Path = Path(os.getenv("RELIMB_INCOMING_DIR", str(PROJECT_ROOT / "data" / "incoming")))
    results_dir: Path = Path(os.getenv("RELIMB_RESULTS_DIR", str(PROJECT_ROOT / "data" / "results")))
    task_timeout_s: int = int(os.getenv("RELIMB_TASK_TIMEOUT_S", "600"))
    fixed_label: str | None = os.getenv("RELIMB_FIXED_LABEL")
    api_host: str = os.getenv("RELIMB_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("RELIMB_API_PORT", "8000"))
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("RELIMB_GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


settings = Settings()

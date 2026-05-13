from __future__ import annotations

from celery import Celery

from src.core.settings import settings

celery_app = Celery(
    "relimgait",
    broker=settings.redis_url,
    backend=settings.celery_backend_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
)

celery_app.autodiscover_tasks(["src.worker"])

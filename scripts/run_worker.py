from __future__ import annotations

from src.worker.celery_app import celery_app


def main() -> None:
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--pool=solo",
        "--concurrency=1",
    ])


if __name__ == "__main__":
    main()

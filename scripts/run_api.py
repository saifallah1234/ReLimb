from __future__ import annotations

import uvicorn

from src.core.settings import settings


def main() -> None:
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()

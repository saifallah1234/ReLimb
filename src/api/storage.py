from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


def save_upload(upload: UploadFile, dest_dir: Path) -> tuple[Path, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    original_name = Path(upload.filename or "video")
    suffix = original_name.suffix or ".mp4"
    video_id = f"{original_name.stem}-{uuid4().hex[:8]}"
    target_path = dest_dir / f"{video_id}{suffix}"

    with target_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return target_path, video_id

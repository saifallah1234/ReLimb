from __future__ import annotations

from pathlib import Path

from src.pose_extraction.normalize_poses import normalize_session, SESSION_DIR


def main() -> None:
    sessions = [p.name for p in SESSION_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(sessions)} session folders")

    ok = 0
    for sid in sessions:
        try:
            normalize_session(sid)
            ok += 1
        except Exception as e:
            print(f"❌ Failed {sid}: {e}")

    print(f"Done. Normalized {ok}/{len(sessions)}")


if __name__ == "__main__":
    main()

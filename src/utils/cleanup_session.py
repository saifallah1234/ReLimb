import os
import shutil
import pandas as pd
from pathlib import Path

# ---------- CONFIGURATION ----------
SESSION_DIR = Path('data/sessions')

def purge_invalid_sessions():
    """
    Deletes session folders if the CSV is unreadable or has no data rows.
    """
    if not SESSION_DIR.exists():
        print("❌ Session directory not found.")
        return

    sessions = sorted([d for d in SESSION_DIR.iterdir() if d.is_dir()])
    deleted_count = 0
    total_count = len(sessions)

    print(f"🔍 Scanning {total_count} sessions for invalid gait data...")

    for session_path in sessions:
        ev_path = session_path / "detected_events.csv"
        
        should_delete = False
        
        # 1. If file doesn't exist at all
        if not ev_path.exists():
            should_delete = True
        
        # 2. Check if file is logically empty
        else:
            try:
                # Read the CSV
                df = pd.read_csv(ev_path)
                # If there are no rows of data, it's useless
                if df.empty:
                    should_delete = True
            except Exception:
                # If pandas can't even parse it (No columns to parse error), it's junk
                should_delete = True

        if should_delete:
            try:
                shutil.rmtree(session_path)
                deleted_count += 1
                print(f"  🗑️ Deleted junk folder: {session_path.name}")
            except Exception as e:
                print(f"  ❌ Error deleting {session_path.name}: {e}")

    print("-" * 30)
    print(f"✅ Cleanup finished.")
    print(f"Total folders scanned: {total_count}")
    print(f"Total folders deleted: {deleted_count}")

if __name__ == "__main__":
    purge_invalid_sessions()
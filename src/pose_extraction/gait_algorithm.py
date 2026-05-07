import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# ---------- CONFIGURATION ----------
SESSION_DIR = Path('data/sessions')

def calculate_asymmetry(left_vals, right_vals):
    l_mean = np.mean(left_vals) if len(left_vals) > 0 else 0
    r_mean = np.mean(right_vals) if len(right_vals) > 0 else 0
    if (l_mean + r_mean) == 0: return 0.0
    return float(abs(l_mean - r_mean) / ((l_mean + r_mean) / 2))

def compute_session_metrics(session_id):
    session_path = SESSION_DIR / session_id
    kp_path = session_path / "keypoints.npy"
    ev_path = session_path / "detected_events.csv"
    output_path = session_path / "gait_metrics.json"

    # 1. Skip if already processed (Resume Logic)
    if output_path.exists():
        return "skipped"

    if not (kp_path.exists() and ev_path.exists()):
        return "missing"

    # 2. Check if CSV is empty before reading
    if os.stat(ev_path).st_size == 0:
        print(f"  ⚠️ Skipping {session_id}: No gait events detected.")
        return "empty"

    try:
        kp = np.load(kp_path)
        df = pd.read_csv(ev_path)
        
        # If the CSV exists but has no data rows
        if df.empty:
            print(f"  ⚠️ Skipping {session_id}: Detected events file is empty.")
            return "empty"

        fps = 25.0 

        hs_l = df[(df.side == 'left') & (df.event == 'heel_strike')].time_s.values
        hs_r = df[(df.side == 'right') & (df.event == 'heel_strike')].time_s.values
        
        stride_times_l = np.diff(hs_l) if len(hs_l) > 1 else []
        stride_times_r = np.diff(hs_r) if len(hs_r) > 1 else []
        
        total_steps = len(hs_l) + len(hs_r)
        duration = len(kp) / fps
        cadence = (total_steps / (duration / 60)) if duration > 0 else 0

        step_lengths = []
        for _, row in df[df.event == 'heel_strike'].iterrows():
            f = int(row.frame)
            if f >= len(kp): continue
            # Landmarks 31/32 are foot indices
            dist = abs(kp[f, 31*2] - kp[f, 32*2])
            step_lengths.append(dist)

        metrics = {
            "session_info": {
                "session_id": session_id,
                "total_frames": len(kp),
                "duration_sec": round(duration, 2)
            },
            "gait_cycle": {
                "cadence_bpm": round(cadence, 2),
                "step_count": total_steps,
                "stride_time_avg_l": round(float(np.mean(stride_times_l)), 3) if len(stride_times_l) > 0 else 0,
                "stride_time_avg_r": round(float(np.mean(stride_times_r)), 3) if len(stride_times_r) > 0 else 0,
            },
            "symmetry": {
                "stride_time_asymmetry": round(calculate_asymmetry(stride_times_l, stride_times_r), 4),
                "step_length_pixel_avg": round(float(np.mean(step_lengths)), 2) if step_lengths else 0
            }
        }

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
        return "success"

    except Exception as e:
        print(f"  ❌ Error in {session_id}: {e}")
        return "error"

def main():
    sessions = sorted([d.name for d in SESSION_DIR.iterdir() if d.is_dir() and "clip" in d.name.lower()])
    
    print(f"Found {len(sessions)} CLIPPED sessions to analyze.")
    
    counts = {"success": 0, "skipped": 0, "empty": 0, "missing": 0, "error": 0}

    for s_id in sessions:
        result = compute_session_metrics(s_id)
        counts[result] += 1
        if result == "success":
            print(f"  ✅ Metrics calculated: {s_id}")

    print(f"\nProcessing Complete!")
    print(f"Success: {counts['success']} | Skipped: {counts['skipped']} | Empty/Invalid: {counts['empty']}")

if __name__ == "__main__":
    main()
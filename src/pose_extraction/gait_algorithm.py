import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------- CONFIGURATION ----------
SESSION_DIR = Path('data/sessions')

def calculate_asymmetry(left_vals, right_vals):
    """Calculates the asymmetry index between two sides."""
    l_mean = np.mean(left_vals) if len(left_vals) > 0 else 0
    r_mean = np.mean(right_vals) if len(right_vals) > 0 else 0
    if (l_mean + r_mean) == 0: return 0.0
    return float(abs(l_mean - r_mean) / ((l_mean + r_mean) / 2))

def compute_session_metrics(session_id):
    session_path = SESSION_DIR / session_id
    kp_path = session_path / "keypoints.npy"
    ev_path = session_path / "detected_events.csv"

    if not (kp_path.exists() and ev_path.exists()):
        print(f"  ⚠ Missing data for {session_id}")
        return

    # Load data
    kp = np.load(kp_path)  # [N, 66]
    df = pd.read_csv(ev_path)
    fps = 25.0  # Convention for this project

    # 1. Temporal Analysis (Timing)
    hs_l = df[(df.side == 'left') & (df.event == 'heel_strike')].time_s.values
    hs_r = df[(df.side == 'right') & (df.event == 'heel_strike')].time_s.values
    
    stride_times_l = np.diff(hs_l) if len(hs_l) > 1 else []
    stride_times_r = np.diff(hs_r) if len(hs_r) > 1 else []

    # 2. Cadence (Steps per minute)
    total_steps = len(hs_l) + len(hs_r)
    duration = len(kp) / fps
    cadence = (total_steps / (duration / 60)) if duration > 0 else 0

    # 3. Spatial Analysis (Step Length in Pixels)
    # We measure the X-distance between feet exactly at the moment of Heel Strike
    step_lengths = []
    for _, row in df[df.event == 'heel_strike'].iterrows():
        f = int(row.frame)
        if f >= len(kp): continue
        # Distance between Left Foot Index (31) and Right Foot Index (32)
        dist = abs(kp[f, 31*2] - kp[f, 32*2])
        step_lengths.append(dist)

    # 4. Compile Metrics
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

    # Save to JSON
    with open(session_path / "gait_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"  ✅ Metrics calculated: Cadence={metrics['gait_cycle']['cadence_bpm']}")

def main():
    # Process every session folder found in data/sessions
    sessions = [d.name for d in SESSION_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(sessions)} sessions to analyze.")
    
    for s_id in sessions:
        print(f"▶ Analyzing: {s_id}")
        compute_session_metrics(s_id)

if __name__ == "__main__":
    main()
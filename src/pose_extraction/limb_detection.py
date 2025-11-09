import pandas as pd
import os

# === PATH CONFIG ===
SESSION_DIR = "data/sessions"
OUTPUT_FEATURES_CSV = "src/features/feature_events.xlsx"  # Single global CSV file

# === COMPUTE BASIC METRICS ===
def compute_gait_times(events):
    heel_strikes = events[events['event'] == 'heel_strike']['time_s'].values
    toe_offs = events[events['event'] == 'toe_off']['time_s'].values
    
    stride_times, stance_times, swing_times = [], [], []
    
    for i in range(len(heel_strikes) - 1):
        stride_times.append(heel_strikes[i + 1] - heel_strikes[i])
        
        # Find nearest toe off after heel strike
        later_toe_offs = toe_offs[toe_offs > heel_strikes[i]]
        if len(later_toe_offs) > 0:
            stance_times.append(later_toe_offs[0] - heel_strikes[i])
        
        # Find next heel strike after toe off for swing time
        later_hs = heel_strikes[heel_strikes > later_toe_offs[0]] if len(later_toe_offs) > 0 else []
        if len(later_hs) > 0:
            swing_times.append(later_hs[0] - later_toe_offs[0])
    
    return {
        "stride_mean": sum(stride_times)/len(stride_times) if stride_times else 0,
        "stance_mean": sum(stance_times)/len(stance_times) if stance_times else 0,
        "swing_mean": sum(swing_times)/len(swing_times) if swing_times else 0,
        "stride_times": stride_times,
        "stance_times": stance_times,
        "swing_times": swing_times
    }

# === ASYMMETRY COMPUTATION ===
def compute_asymmetry(events_path, session_id):
    df = pd.read_csv(events_path)
    left = df[df['side'] == 'left']
    right = df[df['side'] == 'right']
    
    left_metrics = compute_gait_times(left)
    right_metrics = compute_gait_times(right)

    # Gait asymmetry index
    def pct_diff(a, b):
        return abs(a - b) / ((a + b) / 2) * 100 if (a + b) != 0 else 0

    stride_asymmetry = pct_diff(left_metrics["stride_mean"], right_metrics["stride_mean"])
    stance_asymmetry = pct_diff(left_metrics["stance_mean"], right_metrics["stance_mean"])
    swing_asymmetry = pct_diff(left_metrics["swing_mean"], right_metrics["swing_mean"])
    
    return {
        "session_id": session_id,
        "Left stride mean": left_metrics['stride_mean'],
        "Right stride mean": right_metrics['stride_mean'],
        "Left stance mean": left_metrics['stance_mean'],
        "Right stance mean": right_metrics['stance_mean'],
        "Left swing mean": left_metrics['swing_mean'],
        "Right swing mean": right_metrics['swing_mean'],
        "Asymmetry (stride)": stride_asymmetry,
        "Asymmetry (stance)": stance_asymmetry,
        "Asymmetry (swing)": swing_asymmetry
    }

# === MAIN LOOP ===
def main():
    print(f"🔍 Scanning session folders in: {SESSION_DIR}")
    
    session_folders = [
        os.path.join(SESSION_DIR, d)
        for d in os.listdir(SESSION_DIR)
        if os.path.isdir(os.path.join(SESSION_DIR, d))
    ]
    
    if not session_folders:
        print("⚠️ No session folders found!")
        return

    all_rows = []

    for session_path in session_folders:
        events_path = os.path.join(session_path, "detected_events.csv")
        if not os.path.exists(events_path):
            print(f"⚠️ No detected_events.csv found in {session_path}")
            continue
        
        session_id = os.path.basename(session_path)
        print(f"▶ Processing session: {session_id}")
        
        try:
            row = compute_asymmetry(events_path, session_id)
            all_rows.append(row)
        except Exception as e:
            print(f"❌ Error processing {session_id}: {e}")
    
    # Combine all rows into one DataFrame
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)
        df_all.to_excel(OUTPUT_FEATURES_CSV, index=False)
        print(f"💾 Saved combined feature dataset to {OUTPUT_FEATURES_CSV}")
    else:
        print("⚠️ No valid sessions processed.")

# === RUN ===
if __name__ == "__main__":
    main()

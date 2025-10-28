import pandas as pd
import numpy as np

# CONFIG (tune for your cohort)
PCT_THRESHOLD = 15.0     # percent difference considered clinically relevant
ABS_THRESHOLD = 0.10     # absolute seconds difference considered relevant
MIN_SAMPLES = 1          # minimum samples to compute means

# Load your CSV
df = pd.read_csv('data/detected_events.csv')

# Separate sides
left = df[df['side'] == 'left']
right = df[df['side'] == 'right']

def safe_mean(lst):
    if len(lst) == 0:
        return None
    return float(np.mean(lst))

# Helper function to compute stride, stance, swing
def compute_gait_times(events):
    heel_strikes = events[events['event'] == 'heel_strike']['time_s'].values
    toe_offs = events[events['event'] == 'toe_off']['time_s'].values
    
    stride_times = []
    stance_times = []
    swing_times = []
    
    # ensure arrays sorted
    heel_strikes = np.sort(heel_strikes)
    toe_offs = np.sort(toe_offs)
    
    for i in range(len(heel_strikes) - 1):
        # stride time (same foot to next same foot)
        s = heel_strikes[i + 1] - heel_strikes[i]
        stride_times.append(s)
        
        # first toe-off that occurs AFTER the current heel strike
        later_toe_offs = toe_offs[toe_offs > heel_strikes[i]]
        if len(later_toe_offs) > 0:
            stance = later_toe_offs[0] - heel_strikes[i]
            stance_times.append(stance)
            # swing = next heel after that toe-off (same foot)
            later_hs = heel_strikes[heel_strikes > later_toe_offs[0]]
            if len(later_hs) > 0:
                swing = later_hs[0] - later_toe_offs[0]
                swing_times.append(swing)
    
    return {
        "stride_mean": safe_mean(stride_times),
        "stance_mean": safe_mean(stance_times),
        "swing_mean": safe_mean(swing_times),
        "stride_times": stride_times,
        "stance_times": stance_times,
        "swing_times": swing_times
    }

left_metrics = compute_gait_times(left)
right_metrics = compute_gait_times(right)

# Helper to compute diffs and percent
def comp(a, b):
    """Return (a_mean, b_mean, abs_diff, pct_diff) where pct_diff is (abs(a-b)/mean(a,b))*100.
       If either None, returns None values accordingly."""
    if a is None or b is None:
        return (a, b, None, None)
    abs_diff = abs(a - b)
    mean_ab = (a + b) / 2.0 if (a + b) != 0 else 0.0
    pct_diff = (abs_diff / mean_ab) * 100.0 if mean_ab != 0 else 0.0
    return (a, b, abs_diff, pct_diff)

# Compare metrics
stride_a, stride_b, stride_abs, stride_pct = comp(left_metrics["stride_mean"], right_metrics["stride_mean"])
stance_a, stance_b, stance_abs, stance_pct = comp(left_metrics["stance_mean"], right_metrics["stance_mean"])
swing_a, swing_b, swing_abs, swing_pct = comp(left_metrics["swing_mean"], right_metrics["swing_mean"])

# Basic limp detection (example rule)
limp_detected = None
if stride_a is not None and stride_b is not None:
    # If one stride mean is much smaller than the other (e.g., < 70% of other) → limp on that side
    if stride_b < 0.7 * stride_a:
        limp_detected = "RIGHT limb may be underloading (shorter stride)"
    elif stride_a < 0.7 * stride_b:
        limp_detected = "LEFT limb may be underloading (shorter stride)"
    else:
        limp_detected = "No strong limp detected by stride rule"

# Interpret asymmetries with human-readable messages
interpretations = []

# Stance interpretation: shorter stance -> offloading (can't bear weight) on that leg
if stance_abs is not None:
    if (stance_pct >= PCT_THRESHOLD) or (stance_abs >= ABS_THRESHOLD):
        if stance_b < stance_a:
            interpretations.append(f"Right stance mean is shorter ({stance_b:.2f}s) vs Left ({stance_a:.2f}s). "
                                   "This suggests the right leg is spending less time in stance (possible weight offloading / pain / poor prosthetic fit).")
        elif stance_a < stance_b:
            interpretations.append(f"Left stance mean is shorter ({stance_a:.2f}s) vs Right ({stance_b:.2f}s). "
                                   "This suggests the left leg is spending less time in stance (possible weight offloading / pain / poor prosthetic fit).")
    else:
        interpretations.append("Stance times are roughly symmetric (no clinically-relevant difference detected).")
else:
    interpretations.append("Insufficient stance data to compare.")

# Swing interpretation: shorter swing time -> faster swing phase (leg moves quicker)
if swing_abs is not None:
    if (swing_pct >= PCT_THRESHOLD) or (swing_abs >= ABS_THRESHOLD):
        if swing_b < swing_a:
            interpretations.append(f"Right swing mean is shorter ({swing_b:.2f}s) vs Left ({swing_a:.2f}s). "
                                   "This means the right leg swings faster (shorter swing duration) — could indicate quicker transfer or compensation.")
        elif swing_a < swing_b:
            interpretations.append(f"Left swing mean is shorter ({swing_a:.2f}s) vs Right ({swing_b:.2f}s). "
                                   "This means the left leg swings faster (shorter swing duration).")
    else:
        interpretations.append("Swing times are roughly symmetric (no clinically-relevant difference detected).")
else:
    interpretations.append("Insufficient swing data to compare.")

# Stride interpretation (asymmetry in stride length/time)
if stride_abs is not None:
    if (stride_pct >= PCT_THRESHOLD) or (stride_abs >= ABS_THRESHOLD):
        if stride_b < stride_a:
            interpretations.append(f"Right stride mean is shorter ({stride_b:.2f}s) vs Left ({stride_a:.2f}s). "
                                   "Likely shorter steps on the right side (may indicate limp or reduced push-off).")
        elif stride_a < stride_b:
            interpretations.append(f"Left stride mean is shorter ({stride_a:.2f}s) vs Right ({stride_b:.2f}s).")
    else:
        interpretations.append("Stride times are roughly symmetric (no clinically-relevant difference detected).")
else:
    interpretations.append("Insufficient stride data to compare.")

# Print a neat report
print("=== GAIT ANALYSIS RESULTS ===")
def fmt_val(v): return f"{v:.2f}s" if v is not None else "N/A"
print(f"Left stride mean : {fmt_val(left_metrics['stride_mean'])}")
print(f"Right stride mean: {fmt_val(right_metrics['stride_mean'])}")
print(f"Left stance mean : {fmt_val(left_metrics['stance_mean'])}")
print(f"Right stance mean: {fmt_val(right_metrics['stance_mean'])}")
print(f"Left swing mean  : {fmt_val(left_metrics['swing_mean'])}")
print(f"Right swing mean : {fmt_val(right_metrics['swing_mean'])}")
print()
if stride_abs is not None:
    print(f"Stride difference: {stride_abs:.3f}s ({stride_pct:.1f}%)")
if stance_abs is not None:
    print(f"Stance  difference: {stance_abs:.3f}s ({stance_pct:.1f}%)")
if swing_abs is not None:
    print(f"Swing   difference: {swing_abs:.3f}s ({swing_pct:.1f}%)")
print()
print("Limp rule result:", limp_detected)
print("\nInterpretations:")
for s in interpretations:
    print("-", s)

# Optionally: save a short JSON/csv result (uncomment if needed)
# result = {
#     "left_stride_mean": left_metrics['stride_mean'],
#     "right_stride_mean": right_metrics['stride_mean'],
#     "left_stance_mean": left_metrics['stance_mean'],
#     "right_stance_mean": right_metrics['stance_mean'],
#     "left_swing_mean": left_metrics['swing_mean'],
#     "right_swing_mean": right_metrics['swing_mean'],
#     "stride_diff_s": stride_abs, "stride_diff_pct": stride_pct,
#     "stance_diff_s": stance_abs, "stance_diff_pct": stance_pct,
#     "swing_diff_s": swing_abs, "swing_diff_pct": swing_pct,
#     "limp_detected": limp_detected
# }
# pd.DataFrame([result]).to_csv('data/gait_summary.csv', index=False)

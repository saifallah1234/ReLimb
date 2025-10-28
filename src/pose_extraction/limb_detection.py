import pandas as pd

# Load your CSV
df = pd.read_csv('data/detected_events.csv')

# Separate sides
left = df[df['side'] == 'left']
right = df[df['side'] == 'right']

# Helper function to compute stride, stance, swing
def compute_gait_times(events):
    heel_strikes = events[events['event'] == 'heel_strike']['time_s'].values
    toe_offs = events[events['event'] == 'toe_off']['time_s'].values
    
    stride_times = []
    stance_times = []
    swing_times = []
    
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
        "stride_mean": sum(stride_times)/len(stride_times),
        "stance_mean": sum(stance_times)/len(stance_times),
        "swing_mean": sum(swing_times)/len(swing_times),
        "stride_times": stride_times,
        "stance_times": stance_times,
        "swing_times": swing_times
    }

left_metrics = compute_gait_times(left)
right_metrics = compute_gait_times(right)

# Compute step times (time between opposite heel strikes)
step_times = []
for i in range(min(len(left[left['event'] == 'heel_strike']), len(right[right['event'] == 'heel_strike'])) - 1):
    step_times.append(abs(left[left['event'] == 'heel_strike']['time_s'].values[i] -
                          right[right['event'] == 'heel_strike']['time_s'].values[i]))

# Gait asymmetry index
stride_asymmetry = abs(left_metrics["stride_mean"] - right_metrics["stride_mean"]) / ((left_metrics["stride_mean"] + right_metrics["stride_mean"]) / 2) * 100
stance_asymmetry = abs(left_metrics["stance_mean"] - right_metrics["stance_mean"]) / ((left_metrics["stance_mean"] + right_metrics["stance_mean"]) / 2) * 100
swing_asymmetry = abs(left_metrics["swing_mean"] - right_metrics["swing_mean"]) / ((left_metrics["swing_mean"] + right_metrics["swing_mean"]) / 2) * 100

# Limp detection (basic heuristic)
limp = right_metrics["stride_mean"] < 0.7 * left_metrics["stride_mean"]

print("=== GAIT ANALYSIS RESULTS ===")
print(f"Left stride mean: {left_metrics['stride_mean']:.2f}s")
print(f"Right stride mean: {right_metrics['stride_mean']:.2f}s")
print(f"Left stance mean: {left_metrics['stance_mean']:.2f}s")
print(f"Right stance mean: {right_metrics['stance_mean']:.2f}s")
print(f"left swing mean: {right_metrics['swing_mean']:.2f}s")
print(f"Right swing mean: {right_metrics['swing_mean']:.2f}s")
print(f"Asymmetry (stride): {stride_asymmetry:.1f}%")
print(f"Asymmetry (stance): {stance_asymmetry:.1f}%")
print(f"Asymmetry (swing): {swing_asymmetry:.1f}%")
print(f"Limp detected: {'YES' if limp else 'NO'}")

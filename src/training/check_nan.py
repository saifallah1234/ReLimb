"""
check_nan_pattern.py
Checks whether NaN frames are clustered at clip boundaries or random scatter.
"""
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_DIR  = PROJECT_ROOT / "data" / "sessions"

clip_folders = sorted([
    f for f in SESSION_DIR.iterdir()
    if f.is_dir() and "_clip_" in f.name and (f / "keypoints.npy").exists()
])

# Sample 200 clips
sample = clip_folders[:200]

start_nan   = []   # NaN frames in first 30 frames
end_nan     = []   # NaN frames in last 30 frames
middle_nan  = []   # NaN frames in middle 90 frames
total_nan   = []
all_zero_clips = 0

for folder in sample:
    kp = np.load(str(folder / "keypoints.npy"))[:150]
    if kp.shape[0] < 150:
        pad = np.zeros((150 - kp.shape[0], kp.shape[1]))
        kp  = np.concatenate([kp, pad], axis=0)

    nan_mask = np.isnan(kp).any(axis=1)   # [150] bool
    total_nan.append(nan_mask.sum())
    start_nan.append(nan_mask[:30].sum())
    end_nan.append(nan_mask[30:120].sum())
    middle_nan.append(nan_mask[120:].sum())

    if nan_mask.sum() == 150:
        all_zero_clips += 1

print("── NaN frame distribution across 200 sampled clips ──")
print(f"  Mean NaN frames per clip    : {np.mean(total_nan):.1f} / 150")
print(f"  Clips with 0 NaN frames     : {sum(n==0 for n in total_nan)}")
print(f"  Clips with >50% NaN frames  : {sum(n>75 for n in total_nan)}")
print(f"  Clips with 100% NaN (empty) : {all_zero_clips}")
print()
print(f"  Mean NaN in first 30 frames  : {np.mean(start_nan):.1f}")
print(f"  Mean NaN in middle 90 frames : {np.mean(end_nan):.1f}")
print(f"  Mean NaN in last 30 frames   : {np.mean(middle_nan):.1f}")
print()

# Verdict
start_ratio  = np.mean(start_nan)  / 30
middle_ratio = np.mean(end_nan)    / 90
end_ratio    = np.mean(middle_nan) / 30

print("── Verdict ──────────────────────────────────────────")
if start_ratio > middle_ratio * 2:
    print("  ⚠ NaN clustered at START — person enters frame late")
    print("    Fix: trim first N NaN frames before feeding LSTM")
elif end_ratio > middle_ratio * 2:
    print("  ⚠ NaN clustered at END — clip extends past the walk")
    print("    Fix: trim trailing NaN frames")
elif np.mean(total_nan) < 15:
    print("  ✓ NaN scatter is low (<10%) — acceptable for training")
elif all_zero_clips > 10:
    print(f"  ⚠ {all_zero_clips} completely empty clips — remove from dataset")
else:
    print("  ~ NaN is scattered across clip (random detection failures)")
    print("    The LSTM handles this reasonably via nan_to_num=0")
    print("    Consider masking zero frames in the LSTM mean-pool")
"""
PATCH INSTRUCTIONS FOR dataset.py
==================================
The model collapses to one class because WeightedRandomSampler is built
from labels AFTER the dataset is constructed, but the dataset's internal
class_mapping may not match the integer IDs the sampler uses.

Apply these three fixes to your ProGaitDataset class.
"""

# ── FIX 1: NaN keypoints ──────────────────────────────────────────────────
# In _load_keypoints() or wherever you load keypoints.npy, add normalization
# AFTER nan_to_num so zero-filled NaN frames don't dominate the LSTM signal.
#
# BEFORE:
#   kp = np.nan_to_num(np.load(path), nan=0.0)
#
# AFTER:
def load_and_normalize_keypoints(path, clip_frames=150):
    import numpy as np

    GAIT_LM = [0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    L_HIP, R_HIP = 23, 24

    kp = np.load(str(path)).astype(np.float32)

    # Pad or trim to fixed length
    if kp.shape[0] < clip_frames:
        pad = np.zeros((clip_frames - kp.shape[0], kp.shape[1]), dtype=np.float32)
        kp  = np.concatenate([kp, pad], axis=0)
    kp = kp[:clip_frames]

    # Hip-center BEFORE nan_to_num so NaN hips don't produce fake zeros
    mid_hip_x = (kp[:, L_HIP*2]   + kp[:, R_HIP*2])   / 2
    mid_hip_y = (kp[:, L_HIP*2+1] + kp[:, R_HIP*2+1]) / 2

    out = np.zeros((clip_frames, len(GAIT_LM) * 2), dtype=np.float32)
    for i, lm in enumerate(GAIT_LM):
        out[:, i*2]   = kp[:, lm*2]   - mid_hip_x
        out[:, i*2+1] = kp[:, lm*2+1] - mid_hip_y

    # NOW replace NaN (frames where hips were invisible → 0 displacement)
    out = np.nan_to_num(out, nan=0.0)

    # Scale to [-1, 1] per clip so pixel scale doesn't dominate
    max_abs = np.abs(out).max()
    if max_abs > 1e-6:
        out /= max_abs

    return out   # [150, 26]


# ── FIX 2: Expose integer labels from dataset for sampler ─────────────────
# The WeightedRandomSampler in train_lstm_only.py calls:
#
#   label_ids = [dataset.class_mapping[dataset.valid_sessions[i]["issue_text"]]
#                for i in train_idx]
#
# This only works if valid_sessions[i]["issue_text"] exactly matches a key
# in class_mapping. Add a property to ProGaitDataset that returns the
# pre-computed integer label for each sample:
#
# In ProGaitDataset.__init__, after building self.valid_sessions, add:
#
#   self.labels = [
#       self.class_mapping.get(s["issue_text"], 0)
#       for s in self.valid_sessions
#   ]
#
# Then in train_lstm_only.py replace:
#
#   label_ids = [dataset.class_mapping[dataset.valid_sessions[i]["issue_text"]]
#                for i in train_idx]
#
# with:
#
#   label_ids = [dataset.labels[i] for i in train_idx]
#
# This removes the dict lookup that silently returns 0 for unknown keys.


# ── FIX 3: Verify sampler is actually used ────────────────────────────────
# The most common reason WeightedRandomSampler fails silently is that
# shuffle=True overrides it. PyTorch raises no error — it just ignores sampler.
# Confirm your DataLoader looks like this (shuffle must be False when sampler is set):

def make_train_loader_correctly(dataset, indices, batch_size, labels):
    """Correct DataLoader construction with WeightedRandomSampler."""
    from collections import Counter
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    counts  = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,       # ← MUST be False when using sampler
        sampler=sampler,
        collate_fn=None,     # replace with your pad_collate_fn
    )


# ── FIX 4: Print per-batch class distribution to verify sampler works ─────
# Add this to your first training epoch to confirm sampler is balanced:

def verify_sampler(loader, num_classes, n_batches=5):
    """Call this at the start of training to confirm sampler is balanced."""
    from collections import Counter
    print("  Verifying sampler — first 5 batch class distributions:")
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        _, _, _, issue, _ = batch
        dist = Counter(issue.tolist())
        print(f"    batch {i+1}: {dict(sorted(dist.items()))}")


# ── QUICK DIAGNOSTIC: run this standalone to check your dataset ───────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))

    from dataset import ProGaitDataset, pad_collate_fn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    ds = ProGaitDataset()

    # Check label field alignment
    print("Checking label field alignment...")
    mismatches = 0
    for s in ds.valid_sessions[:20]:
        text = s.get("issue_text", "MISSING")
        mapped = ds.class_mapping.get(text, "NOT FOUND")
        if mapped == "NOT FOUND":
            print(f"  ✗ '{text}' not in class_mapping")
            mismatches += 1
    if mismatches == 0:
        print("  ✓ All labels align with class_mapping")

    # Build sampler from ALL labels and check one epoch's distribution
    all_labels = [ds.class_mapping.get(s["issue_text"], 0) for s in ds.valid_sessions]
    counts     = Counter(all_labels)
    weights    = [1.0 / counts[l] for l in all_labels]
    sampler    = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        sampler=sampler, collate_fn=pad_collate_fn)

    print("\nSampler test — class distribution across 10 batches:")
    epoch_dist: Counter = Counter()
    for i, batch in enumerate(loader):
        if i >= 10: break
        _, _, _, issue, _ = batch
        epoch_dist.update(issue.tolist())

    class_names = {v: k for k, v in ds.class_mapping.items()}
    total = sum(epoch_dist.values())
    for idx in sorted(epoch_dist):
        pct = epoch_dist[idx] / total * 100
        print(f"  [{idx}] {class_names.get(idx,'?'):<25} {pct:.1f}%")

    if epoch_dist.most_common(1)[0][1] / total > 0.6:
        print("\n⚠ Sampler NOT working — one class still dominates")
        print("  Check that shuffle=False in DataLoader and sampler is passed correctly")
    else:
        print("\n✓ Sampler working — classes roughly balanced")
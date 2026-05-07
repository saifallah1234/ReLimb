"""
diagnose.py
===========
Run this before training to catch silent bugs:
  1. Class mapping vs actual data distribution
  2. Whether clip session_ids match dataset_index entries
  3. What the model actually predicts after 1 epoch (sanity check)
  4. Tests the ProGaitDataset to ensure clean_keypoints() removes all NaNs.

Usage:
    python src/training/diagnose.py
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_DIR  = PROJECT_ROOT / "data" / "sessions"
INDEX_FILE   = PROJECT_ROOT / "data" / "raw_videos" / "hf" / "dataset_index.json"
MAPPING_FILE = PROJECT_ROOT / "data" / "class_mapping.json"


def check_mapping_vs_data():
    print("\n── 1. CLASS MAPPING ─────────────────────────────────")
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)
    print(f"  Classes in mapping file ({len(mapping)}):")
    for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"    [{idx}] {name}")

    print("\n── 2. DATASET INDEX LABELS ──────────────────────────")
    with open(INDEX_FILE) as f:
        index = json.load(f)
    raw_labels = Counter(e.get("clean_primary_issue", "MISSING") for e in index)
    print(f"  Unique labels in dataset_index.json ({len(raw_labels)}):")
    for label, cnt in sorted(raw_labels.items(), key=lambda x: -x[1]):
        in_mapping = "✓" if label in mapping else "✗ NOT IN MAPPING"
        print(f"    {cnt:4d}x  {label}  {in_mapping}")


def check_session_label_alignment():
    print("\n── 3. SESSION ↔ LABEL ALIGNMENT ─────────────────────")
    with open(INDEX_FILE) as f:
        index = json.load(f)
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    # Build lookup: video stem → label
    stem_to_label = {}
    for entry in index:
        raw_id = entry["ID"].replace(".mp4","").replace(".avi","")
        label  = entry.get("clean_primary_issue", "Unknown / Other")
        stem_to_label[raw_id] = label

    clip_folders = [f for f in SESSION_DIR.iterdir()
                    if f.is_dir() and "_clip_" in f.name]

    matched = unmatched = 0
    label_counts: Counter = Counter()
    unmatched_examples = []

    for folder in clip_folders:
        name = folder.name  
        for prefix in ("inside_", "outside_"):
            if name.startswith(prefix):
                stem_with_clip = name[len(prefix):]
                parts = stem_with_clip.rsplit("_clip_", 1)
                stem  = parts[0]
                break
        else:
            stem = name

        label = stem_to_label.get(stem)
        if label is None:
            unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append((folder.name, stem))
        else:
            matched += 1
            label_counts[label] += 1

    print(f"  Matched   : {matched}")
    print(f"  Unmatched : {unmatched}")
    if unmatched_examples:
        print(f"  Unmatched examples (folder → stem tried):")
        for fn, st in unmatched_examples:
            print(f"    {fn}  →  '{st}'")

    print(f"\n  Clip-level label distribution:")
    total = sum(label_counts.values())
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        mapped = mapping.get(label, "❌ NOT MAPPED")
        pct = cnt / total * 100 if total else 0
        print(f"    {cnt:4d} ({pct:4.1f}%)  [{mapped}]  {label}")


def check_dataset_post_cleaning():
    """
    Instead of checking raw files, this checks the output of ProGaitDataset 
    to ensure the pandas interpolation fix successfully removes all NaNs.
    """
    print("\n── 4. DATASET CLEANING SANITY (NaN check) ───────────")
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
        from dataset import ProGaitDataset
    except ImportError as e:
        print(f"  ⚠ Could not import dataset: {e}")
        return

    ds = ProGaitDataset()
    if len(ds) == 0:
        print("  ⚠ Empty dataset")
        return

    sample_size = min(200, len(ds))
    nan_count = 0
    shapes = Counter()

    for i in range(sample_size):
        # This triggers the clean_keypoints() function in dataset.py
        kp, metrics, ccc, issue = ds[i]
        
        shapes[tuple(kp.shape)] += 1
        if torch.isnan(kp).any():
            nan_count += 1

    print(f"  Checked {sample_size} dataset outputs (post-cleaning):")
    for shape, cnt in shapes.most_common():
        print(f"    shape {shape}  ×{cnt}")
        
    if nan_count > 0:
        print(f"  ⚠ FAILED: {nan_count}/{sample_size} items STILL have NaNs! The pandas fix didn't work.")
    else:
        print(f"  ✓ SUCCESS: 0 NaNs found! The `dataset.py` cleaning is working perfectly.")


def check_model_predictions():
    print("\n── 5. MODEL SANITY (Untrained predictions) ──────────")
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
        from dataset import ProGaitDataset, pad_collate_fn
        from model_lstm import GaitSequenceLSTM
    except ImportError as e:
        print(f"  ⚠ Could not import dataset/model: {e}")
        return

    ds = ProGaitDataset()
    if len(ds) == 0: return

    num_classes = len(ds.class_mapping)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

    device    = torch.device("cpu")
    encoder   = GaitSequenceLSTM().to(device)
    head      = nn.Linear(64, num_classes).to(device)

    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            kp, _, _, issue, lengths = batch
            logits = head(encoder(kp, lengths))
            all_preds.extend(logits.argmax(1).tolist())
            all_true.extend(issue.tolist())

            if len(all_preds) >= 100:  # Just check first 100
                break

    pred_dist = Counter(all_preds)
    true_dist = Counter(all_true)
    class_names = {v: k for k, v in ds.class_mapping.items()}

    print(f"  True vs Pred distribution (first 100 samples using RANDOM weights):")
    for idx in sorted(true_dist):
        print(f"    [{idx}] {class_names.get(idx,'?'):<25} true={true_dist[idx]:<3} pred={pred_dist.get(idx,0)}")

    print("\n  (Note: Because these weights are untrained, clumping to one class is normal here.")
    print("         The real test is if it learns during train_lstm_only.py)")


if __name__ == "__main__":
    check_mapping_vs_data()
    check_session_label_alignment()
    check_dataset_post_cleaning()
    check_model_predictions()
    print("\n── Done ─────────────────────────────────────────────\n")
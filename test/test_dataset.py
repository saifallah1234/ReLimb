import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.dataset import ProGaitDataset, pad_collate_fn


def run_dataset_tests():
    print("\n=== DATASET UNIT TEST START ===")

    # ─────────────────────────────────────────────
    # 1. Load dataset
    # ─────────────────────────────────────────────
    dataset = ProGaitDataset()

    assert len(dataset) > 0, "❌ Dataset is empty!"
    # ─────────────────────────────────────────────
    # 2. Single sample test
    # ─────────────────────────────────────────────
    x, metrics, ccc, issue = dataset[0]

    print("Sample keypoints shape:", x.shape)

    # Shape checks
    assert len(x.shape) == 2, "❌ Keypoints should be 2D (T, Features)"
    assert x.shape[1] == 26, "❌ Wrong feature size (expected 26)"
    assert x.shape[0] == 150, "❌ Clip is not 150 frames"

    # NaN check
    assert not torch.isnan(x).any(), "❌ NaNs found in keypoints"

    # Empty / zero check (IMPORTANT)
    total_sum = torch.sum(torch.abs(x)).item()
    assert total_sum > 0, "❌ Keypoints are empty (all zeros)"

    # Value range sanity
    print("Min value:", x.min().item())
    print("Max value:", x.max().item())
    assert x.max() < 1000, "❌ Values too large (bad scaling?)"

    # ─────────────────────────────────────────────
    # 3. Batch test (DataLoader + padding)
    # ─────────────────────────────────────────────
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    keypoints, metrics, ccc, issues, lengths = next(iter(loader))

    print("\nBatch keypoints shape:", keypoints.shape)

    # Batch shape checks
    assert len(keypoints.shape) == 3, "❌ Batch should be 3D (B, T, F)"
    assert keypoints.shape[1] == 150, "❌ Padding / sequence length broken"

    # Batch NaN check
    assert not torch.isnan(keypoints).any(), "❌ NaNs found in batch"

    # Batch empty check
    batch_sum = torch.sum(torch.abs(keypoints)).item()
    assert batch_sum > 0, "❌ Batch keypoints are empty"

    print("\n=== ✅ ALL DATASET TESTS PASSED ===")


if __name__ == "__main__":
    run_dataset_tests()


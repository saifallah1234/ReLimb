import argparse
from pathlib import Path
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset import ProGaitDataset, pad_collate_fn
from src.models.model_mlp import GaitMetricsMLP


def split_indices(n_items: int, val_split: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_items, generator=g).tolist()
    split = max(1, int(n_items * val_split))
    return idx[split:], idx[:split]   # train, val


LOG1P_INDICES = [0, 1, 5]  # cadence, step_count, step_length


def _transform_metrics(metrics: torch.Tensor, log1p: bool) -> torch.Tensor:
    if log1p:
        metrics = metrics.clone()
        metrics[:, LOG1P_INDICES] = torch.log1p(torch.clamp(metrics[:, LOG1P_INDICES], min=0.0))
    return metrics


def run_training(
    epochs: int,
    lr: float,
    batch_size: int,
    val_split: float,
    seed: int,
    max_batches: int | None,
    log1p: bool,
    clip_z: float,
    early_stop: int,
)-> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ProGaitDataset()
    train_idx, val_idx = split_indices(len(dataset), val_split, seed)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    # -------------------------
    # Normalize metrics (IMPORTANT)
    # -------------------------
    train_metrics = np.stack([
        dataset.valid_sessions[i]["metrics"] for i in train_idx
    ])
    if log1p:
        train_metrics = train_metrics.copy()
        train_metrics[:, LOG1P_INDICES] = np.log1p(np.clip(train_metrics[:, LOG1P_INDICES], 0.0, None))

    mean = torch.tensor(train_metrics.mean(axis=0), dtype=torch.float32, device=device)
    std = torch.tensor(train_metrics.std(axis=0), dtype=torch.float32, device=device)
    std = torch.clamp(std, min=1e-6)

    # -------------------------
    # Normalize target (CCC)
    # -------------------------
    train_ccc = np.array([
        dataset.valid_sessions[i]["ccc_score"] for i in train_idx
    ], dtype=np.float32)
    ccc_mean = torch.tensor(train_ccc.mean(), dtype=torch.float32, device=device)
    ccc_std = torch.tensor(train_ccc.std(), dtype=torch.float32, device=device)
    ccc_std = torch.clamp(ccc_std, min=1e-6)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn
    )

    # -------------------------
    # MODEL (pure regression)
    # -------------------------
    model = GaitMetricsMLP(input_dim=6, hidden_dim=64, output_dim=128).to(device)

    # single regression head (CCC)
    head = nn.Sequential(
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    criterion = nn.SmoothL1Loss()  # more stable than pure MSE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    no_improve = 0

    save_path = Path(__file__).resolve().parents[2] / "data" / "models" / "mlp_ccc.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):

        # ---- train ----
        model.train()
        head.train()

        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            _, metrics, ccc, _, _ = batch

            metrics = _transform_metrics(metrics.to(device), log1p)
            metrics = (metrics - mean) / std
            if clip_z > 0:
                metrics = torch.clamp(metrics, -clip_z, clip_z)
            ccc = ccc.to(device)
            ccc_norm = (ccc - ccc_mean) / ccc_std

            optimizer.zero_grad()

            features = model(metrics)
            preds = head(features).squeeze(1)

            loss = criterion(preds, ccc_norm.squeeze(1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()

            if max_batches and i + 1 >= max_batches:
                break

        # ---- validation ----
        model.eval()
        head.eval()

        val_loss = 0.0
        val_mae = 0.0
        val_targets: list[float] = []
        val_preds: list[float] = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                _, metrics, ccc, _, _ = batch

                metrics = _transform_metrics(metrics.to(device), log1p)
                metrics = (metrics - mean) / std
                if clip_z > 0:
                    metrics = torch.clamp(metrics, -clip_z, clip_z)
                ccc = ccc.to(device)
                ccc_norm = (ccc - ccc_mean) / ccc_std

                preds = head(model(metrics)).squeeze(1)
                val_loss += criterion(preds, ccc_norm.squeeze(1)).item()
                preds_denorm = preds * ccc_std + ccc_mean
                val_mae += torch.mean(torch.abs(preds_denorm - ccc.squeeze(1))).item()
                val_targets.extend(ccc.squeeze(1).tolist())
                val_preds.extend(preds_denorm.tolist())

                if max_batches and i + 1 >= max_batches:
                    break

        val_loss /= max(1, len(val_loader))
        val_mae /= max(1, len(val_loader))

        scheduler.step(val_loss)

        val_mse = mean_squared_error(val_targets, val_preds) if val_targets else 0.0
        val_rmse = val_mse ** 0.5
        val_mae_score = mean_absolute_error(val_targets, val_preds) if val_targets else 0.0
        val_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0.0

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss / max(1, len(train_loader)):.4f} | "
            f"Val {val_loss:.4f} | MAE {val_mae:.3f}"
        )

        if mlflow.active_run():
            mlflow.log_metric("train_loss", train_loss / max(1, len(train_loader)), step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            mlflow.log_metric("val_mae", val_mae_score, step=epoch)
            mlflow.log_metric("val_r2", val_r2, step=epoch)

        # save best
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_metrics = {
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae_score,
                "val_r2": val_r2,
            }
            torch.save(
                {
                    "mlp": model.state_dict(),
                    "head": head.state_dict(),
                    "mean": mean,
                    "std": std,
                    "ccc_mean": ccc_mean,
                    "ccc_std": ccc_std
                },
                save_path
            )
        else:
            no_improve += 1
            if early_stop > 0 and no_improve >= early_stop:
                print(f"Early stop at epoch {epoch} (no val improvement in {early_stop} epochs)")
                break

    print(f"\nBest Val Loss: {best_val:.4f}")
    print(f"Saved → {save_path}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--log1p", action="store_true", help="Apply log1p to skewed metrics")
    parser.add_argument("--clip_z", type=float, default=5.0, help="Clip normalized metrics to +/- Z")
    parser.add_argument("--early_stop", type=int, default=10, help="Early stop patience (epochs)")

    args = parser.parse_args()

    run_training(
        args.epochs,
        args.lr,
        args.batch_size,
        args.val_split,
        args.seed,
        args.max_batches or None,
        args.log1p,
        args.clip_z,
        args.early_stop,
    )
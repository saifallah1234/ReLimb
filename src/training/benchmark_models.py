"""
benchmark.py
============
Runs LSTM-only and LSTM-pretrained training back to back and
prints a side-by-side comparison of macro F1 scores.

Usage
-----
    python src/models/benchmark.py
    python src/models/benchmark.py --epochs 50 --n_splits 5 --batch_size 16
"""

import argparse
from datetime import datetime
from pathlib import Path

import mlflow

from train_lstm_only       import run_training as run_lstm
from train_lstm_pretrained import run_training as run_lstm_pretrained

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _start_mlflow(experiment: str = "ReLimb Benchmark") -> None:
    mlflow.set_tracking_uri(f"file:///{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment(experiment)


def _log(metrics: dict | None) -> None:
    if metrics and mlflow.active_run():
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))


def benchmark(
    epochs:          int,
    lr:              float,
    batch_size:      int,
    n_splits:        int,
    seed:            int,
    label_smoothing: float,
    head_dim:        int,
    dropout:         float,
    freeze_epochs:   int,
) -> None:
    print("\n" + "=" * 55)
    print("  ReLimb Benchmark")
    print(f"  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 55)

    shared = dict(
        epochs=epochs, lr=lr, batch_size=batch_size,
        n_splits=n_splits, seed=seed,
        label_smoothing=label_smoothing,
        head_dim=head_dim, dropout=dropout,
    )

    # ── Run 1: LSTM only ──────────────────────────────────────────────────
    print("\n─── LSTM ONLY ───────────────────────────────────────")
    _start_mlflow()
    with mlflow.start_run(run_name="lstm_only"):
        mlflow.log_params({"model": "lstm_only", **shared})
        metrics_lstm = run_lstm(**shared)
        _log(metrics_lstm)

    # ── Run 2: LSTM pretrained ────────────────────────────────────────────
    print("\n─── LSTM PRETRAINED (freeze→fine-tune) ──────────────")
    _start_mlflow()
    with mlflow.start_run(run_name="lstm_pretrained"):
        mlflow.log_params({"model": "lstm_pretrained", "freeze_epochs": freeze_epochs, **shared})
        metrics_pre = run_lstm_pretrained(**shared, freeze_epochs=freeze_epochs)
        _log(metrics_pre)

    # ── Summary ───────────────────────────────────────────────────────────
    f1_lstm = metrics_lstm.get("val_f1_macro", 0.0) if metrics_lstm else 0.0
    f1_pre  = metrics_pre.get("val_f1_macro",  0.0) if metrics_pre  else 0.0
    winner  = "LSTM Pretrained" if f1_pre > f1_lstm else "LSTM Only"

    print("\n" + "=" * 55)
    print("  BENCHMARK RESULTS  (macro F1 across all folds)")
    print("=" * 55)
    print(f"  LSTM Only        : {f1_lstm:.4f}")
    print(f"  LSTM Pretrained  : {f1_pre:.4f}")
    print(f"  Winner           : {winner}")
    print("=" * 55)
    print(f"\nMLflow runs logged to: {REPO_ROOT / 'mlruns'}")
    print("View with:  mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReLimb model benchmark")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--n_splits",        type=int,   default=5)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--head_dim",        type=int,   default=64)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--freeze_epochs",   type=int,   default=20,
                        help="Epochs to freeze encoder before fine-tuning (pretrained run only)")
    args = parser.parse_args()

    benchmark(
        epochs          = args.epochs,
        lr              = args.lr,
        batch_size      = args.batch_size,
        n_splits        = args.n_splits,
        seed            = args.seed,
        label_smoothing = args.label_smoothing,
        head_dim        = args.head_dim,
        dropout         = args.dropout,
        freeze_epochs   = args.freeze_epochs,
    )
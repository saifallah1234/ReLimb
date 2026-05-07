# Model Training Scripts

This folder contains small training scripts to compare the standalone MLP, standalone LSTM, and a pretrained-LSTM fine-tune.

## Scripts
- `train_mlp_only.py` — Trains a simple MLP on gait metrics only (CCC regression, no class labels).
- `train_lstm_only.py` — Trains the base LSTM on keypoints only.
- `train_lstm_pretrained.py` — Fine-tunes the LSTM initialized from `data/models/lstm_pretrained.pth`.
- `benchmark_models.py` — Runs all three in sequence with the same split.

## Quick run
Use the repo root as your working directory so the dataset paths resolve.

- Single model:
  - `python src/models/train_mlp_only.py --epochs 20`
  - `python src/models/train_lstm_only.py --epochs 20`
  - `python src/models/train_lstm_pretrained.py --epochs 20`

- Benchmark:
  - `python src/models/benchmark_models.py --epochs 10`

## Troubleshooting
- If you store sessions in `data/session`, the dataset loader will automatically fall back.
- If `lstm_pretrained.pth` is missing, the pretrained script will warn and continue with random init.

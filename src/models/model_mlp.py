import torch
import torch.nn as nn
class GaitMetricsMLP(nn.Module):
    def __init__(self, n_metrics=5, hidden_dim=32, output_dim=64, dropout_rate=0.3):
        """
        input_dim = n_metrics * 2 = 10
            → 5 gait metric values  (0.0 if missing)
            → 5 validity flags      (1.0 if present, 0.0 if None/NaN)
 
        output_dim = 64  (fuses with LSTM's 128-dim → concat = 192)
        """
        super().__init__()
        input_dim = n_metrics * 2   # 10
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Dropout sits here — BEFORE the output projection
        # so the fusion layer always receives a clean 64-dim vector
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),   # ← moved inside, not after
        )

        # Residual: skip connection from raw input → output dim
        # Lets a strongly predictive metric (e.g. cadence) reach the
        # fusion layer without being diluted by two transformations
        self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        """
        x : (B, 10)
        Returns: (B, 64)  — clean output, no trailing dropout
        """
        out = self.layer2(self.layer1(x))
        out = out + self.residual_proj(x)
        return out
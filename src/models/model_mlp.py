import torch
import torch.nn as nn


class GaitMetricsMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=64, dropout_rate=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.residual = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        out = self.net(x)
        out = out + self.residual(x)
        return out
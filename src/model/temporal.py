import torch
import torch.nn as nn


class TemporalClassifier(nn.Module):
    def __init__(self, feature_dim: int = 1280, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(256, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1280)
        x = self.proj(x)           # (B, T, 256)
        _, (h, _) = self.lstm(x)   # h: (layers, B, hidden)
        return torch.sigmoid(self.head(h[-1]))  # (B, 1)

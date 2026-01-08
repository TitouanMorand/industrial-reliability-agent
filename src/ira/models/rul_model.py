import torch
import torch.nn as nn


class CNNGRURUL(nn.Module):
    def __init__(self, n_features: int, seq_len: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)       # (B, F, T)
        x = self.cnn(x)
        x = x.transpose(1, 2)       # (B, T, C)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(1)

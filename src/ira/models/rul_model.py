from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGRURUL(nn.Module):
    """
    CNN -> GRU -> temporal aggregation -> heads

    Supports:
      - temporal_aggregation: last | mean | max | attention
      - aux HI head (optional): predict HI in [0,1] with sigmoid

    Forward:
      - if aux_hi=False: returns rul_pred (B,)
      - if aux_hi=True : returns (rul_pred (B,), hi_pred (B,))
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        cnn_channels: int = 32,
        cnn_kernel_size: int = 3,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_size: int = 64,
        head_dropout: float = 0.3,
        temporal_aggregation: str = "last",
        aux_hi: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.temporal_aggregation = temporal_aggregation.lower()
        self.aux_hi = bool(aux_hi)
        self.bidirectional = bool(bidirectional)

        pad = cnn_kernel_size // 2

        # A slightly stronger CNN than the 1-layer baseline (still lightweight).
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=cnn_kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=pad),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        rnn_out_size = gru_hidden_size * (2 if bidirectional else 1)

        if self.temporal_aggregation == "attention":
            self.attn = nn.Sequential(
                nn.Linear(rnn_out_size, rnn_out_size),
                nn.Tanh(),
                nn.Linear(rnn_out_size, 1),
            )
        else:
            self.attn = None

        # Main head: RUL regression
        self.head_rul = nn.Sequential(
            nn.Linear(rnn_out_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, 1),
        )

        # Auxiliary HI head (optional)
        if self.aux_hi:
            self.head_hi = nn.Sequential(
                nn.Linear(rnn_out_size, head_hidden_size),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_size, 1),
            )
        else:
            self.head_hi = None

    def aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, H)
        returns (B, H)
        """
        if self.temporal_aggregation == "last":
            return h[:, -1, :]

        if self.temporal_aggregation == "mean":
            return h.mean(dim=1)

        if self.temporal_aggregation == "max":
            return h.max(dim=1).values

        if self.temporal_aggregation == "attention":
            # scores: (B, T, 1) -> weights: (B, T, 1)
            scores = self.attn(h)
            weights = torch.softmax(scores, dim=1)
            ctx = (weights * h).sum(dim=1)
            return ctx

        raise ValueError("temporal_aggregation must be one of: last|mean|max|attention")

    def forward(self, x: torch.Tensor):
        # x: (B, T, F) -> conv expects (B, F, T)
        x = x.transpose(1, 2)
        z = self.cnn(x)
        # back to (B, T, C)
        z = z.transpose(1, 2)

        h, _ = self.gru(z)
        pooled = self.aggregate(h)

        rul = self.head_rul(pooled).squeeze(-1)

        if not self.aux_hi:
            return rul

        hi_logits = self.head_hi(pooled).squeeze(-1)
        hi = torch.sigmoid(hi_logits)  # HI in [0,1]
        return rul, hi

"""PyTorch networks.

Kept in their own module so pickled models refer to src.ml.nets rather than
whichever script happened to train them.
"""

import torch
from torch import nn


class FightWinnerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)

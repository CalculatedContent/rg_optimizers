"""The common MLP3 architecture used by all baseline runs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    """A 784 -> 512 -> 512 -> 10 ReLU MLP with no dropout or batch norm."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

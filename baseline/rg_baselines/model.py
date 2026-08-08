"""The common MLP3 architecture used by all baseline runs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    """A 784 -> 512 -> 512 -> 10 ReLU MLP with explicit initialization.

    The two hidden layers use fan-in Kaiming initialization for ReLU activations.
    The linear classifier uses Xavier initialization. Biases start at zero. This
    avoids relying on ``nn.Linear``'s generic default, whose variance is smaller
    than the ReLU-preserving He/Kaiming value.
    """

    initialization_name = "kaiming_uniform_relu_hidden_xavier_uniform_head_v1"

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(
                layer.weight,
                a=0.0,
                mode="fan_in",
                nonlinearity="relu",
            )
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1.0)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

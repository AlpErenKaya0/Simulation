from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("Network dimensions must be positive.")

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))

    def save(self, file_name: str = "drone_dqn_weights.pth") -> None:
        model_directory = Path("model")
        model_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), model_directory / file_name)

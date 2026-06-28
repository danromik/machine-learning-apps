"""Network, device picking, and path constants for the Cube RL app.

Single source of truth for the checkpoints path and `pick_device`, mirroring the
other apps in the suite.

There is one network here: `CostToGoNet`, an MLP over the one-hot facelet
encoding that predicts a single scalar — *how many moves from solved* this state
is (the learned heuristic at the heart of value iteration). The cube observation
is always a flat vector, so there is no CNN branch (unlike Snake's grid mode).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CostToGoNet(nn.Module):
    """Maps a one-hot cube state to a single scalar cost-to-go (moves-to-solve).

    A straightforward MLP with a couple of wide hidden layers — enough capacity
    for the 2x2 and to make real progress on the 3x3, while staying small enough
    to train on local hardware. The output is unbounded and non-negative in
    spirit (it regresses move counts); we don't clamp it, letting the network
    learn the floor at 0 for solved states.
    """

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

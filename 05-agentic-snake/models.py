"""Networks, device picking, and path constants for the Snake RL app.

Single source of truth for the checkpoints path and `pick_device`, mirroring
the other apps in the suite. The two deep agents (DQN, REINFORCE) share the
same body over the chosen observation — only the head's meaning differs
(Q-values vs. action logits). The body adapts to the observation shape:

* engineered features (a 1-D vector) -> a tiny two-hidden-layer **MLP**.
* the full grid (a (C, H, W) tensor) -> a small **CNN** so the network can
  exploit the spatial structure of the board (like an image classifier).

Both end in a linear head of `NUM_ACTIONS` outputs, so the agents downstream
don't care which body they got.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from game import NUM_ACTIONS, STATE_SIZE

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mlp(in_features: int, hidden: int, out: int) -> nn.Sequential:
    """A small two-hidden-layer MLP over a flat feature vector."""
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out),
    )


def _cnn(obs_shape: tuple[int, ...], hidden: int, out: int) -> nn.Sequential:
    """A small CNN over a (channels, height, width) grid observation.

    Two padded 3x3 conv layers preserve the board size, so the flattened
    feature count is `32 * height * width` regardless of grid dimensions; a
    fully-connected hidden layer then feeds the action head.
    """
    c, h, w = obs_shape
    return nn.Sequential(
        nn.Conv2d(c, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * h * w, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out),
    )


def _build_net(obs_shape: tuple[int, ...], hidden: int, out: int) -> nn.Sequential:
    """MLP for a 1-D observation, CNN for a 3-D (grid) one."""
    if len(obs_shape) == 1:
        return _mlp(obs_shape[0], hidden, out)
    return _cnn(obs_shape, hidden, out)


class QNetwork(nn.Module):
    """Maps a state to a Q-value per action (used by DQN)."""

    def __init__(self, obs_shape: tuple[int, ...] = (STATE_SIZE,), hidden: int = 128):
        super().__init__()
        self.net = _build_net(tuple(obs_shape), hidden, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """Maps a state to action logits; softmax gives the policy (used by REINFORCE)."""

    def __init__(self, obs_shape: tuple[int, ...] = (STATE_SIZE,), hidden: int = 128):
        super().__init__()
        self.net = _build_net(tuple(obs_shape), hidden, NUM_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

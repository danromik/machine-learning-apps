"""Training session for the Math Symbol Trainer.

Builds a torch model from the user-designed architecture, holds the
optimizer + step counter across requests, and exposes predict /
train-one-batch / save / load operations.

The architecture spec the frontend ships is a list of layer dicts:
    { type: 'conv2d' | 'maxpool2d' | 'flatten' | 'linear' | 'relu' | 'dropout',
      params: { ... } }

A final `nn.Linear → num_classes` is appended implicitly — the user's
"Output" block represents this projection plus the softmax head.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from synthesis import CANVAS_SIZE


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


# ── Model construction ────────────────────────────────────────────────


def build_model(layers: list[dict[str, Any]], num_classes: int) -> nn.Module:
    """Build a torch.nn.Sequential from the architecture layer list."""
    in_shape: tuple[int, ...] = (1, CANVAS_SIZE, CANVAS_SIZE)
    modules: list[nn.Module] = []

    for layer in layers:
        t = layer["type"]
        p = layer.get("params", {}) or {}
        if t == "conv2d":
            ic = in_shape[0]
            oc = int(p["out_channels"])
            k = int(p["kernel"])
            pad = int(p.get("padding", 0))
            s = int(p.get("stride", 1))
            modules.append(nn.Conv2d(ic, oc, k, stride=s, padding=pad))
            in_shape = (
                oc,
                (in_shape[1] + 2 * pad - k) // s + 1,
                (in_shape[2] + 2 * pad - k) // s + 1,
            )
        elif t == "maxpool2d":
            k = int(p["kernel"])
            s = int(p.get("stride", k))
            modules.append(nn.MaxPool2d(k, stride=s))
            in_shape = (
                in_shape[0],
                (in_shape[1] - k) // s + 1,
                (in_shape[2] - k) // s + 1,
            )
        elif t == "flatten":
            modules.append(nn.Flatten())
            total = 1
            for d in in_shape:
                total *= d
            in_shape = (total,)
        elif t == "linear":
            ic = in_shape[0]
            oc = int(p["out_features"])
            modules.append(nn.Linear(ic, oc))
            in_shape = (oc,)
        elif t == "relu":
            modules.append(nn.ReLU())
        elif t == "dropout":
            modules.append(nn.Dropout(float(p.get("p", 0.25))))
        else:
            raise ValueError(f"Unknown layer type: {t}")

    if len(in_shape) != 1:
        raise ValueError(
            f"Architecture must end with a 1-D shape (add Flatten + Linear "
            f"before Output); got {in_shape}"
        )

    # Implicit final classifier head.
    modules.append(nn.Linear(in_shape[0], num_classes))
    return nn.Sequential(*modules)


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


# ── Image decoding ────────────────────────────────────────────────────


def _decode_image(png_b64: str) -> torch.Tensor:
    """Decode a base64 PNG into a (1, H, W) float32 tensor in [0, 1]."""
    raw = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    # bytearray gives a writable buffer; frombuffer on bytes() warns about
    # non-writability since the underlying memory is shared.
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return arr.reshape(1, img.height, img.width).to(torch.float32) / 255.0


def _decode_batch(images: list[str], device: torch.device) -> torch.Tensor:
    return torch.stack([_decode_image(b) for b in images]).to(device)


# ── Training session ──────────────────────────────────────────────────


class TrainingSession:
    """Holds the model + optimizer + class-index map across requests."""

    def __init__(
        self,
        layers: list[dict[str, Any]],
        hyperparameters: dict[str, Any],
        classes: list[str],
        device: torch.device,
    ) -> None:
        if not classes:
            raise ValueError("classes list is empty")
        self.layers = layers
        self.hyperparameters = dict(hyperparameters)
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.label_to_index = {c: i for i, c in enumerate(self.classes)}
        self.device = device

        self.lr = float(hyperparameters["lr"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.optimizer_name = str(hyperparameters["optimizer"])

        self.model = build_model(layers, self.num_classes).to(device)
        self.optimizer = make_optimizer(
            self.optimizer_name, self.model.parameters(), self.lr
        )
        self.step = 0

    # ── inference / training ──────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images: list[str]) -> list[list[float]]:
        """Forward-only. Returns softmax probs per image, shape (N, num_classes)."""
        if not images:
            return []
        self.model.eval()
        x = _decode_batch(images, self.device)
        probs = F.softmax(self.model(x), dim=1)
        return probs.cpu().tolist()

    def set_lr(self, lr: float) -> None:
        """Hot-swap the optimizer's learning rate without rebuilding it.

        Optimizer state (Adam moments, SGD momentum) is preserved — only
        the lr field on each param_group is rewritten.
        """
        lr = float(lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.lr = lr

    def train_batch(
        self,
        images: list[str],
        labels: list[str],
        lr: float | None = None,
        optimizer_name: str | None = None,
    ) -> dict[str, Any]:
        """One forward + backward + optimizer step. Returns the loss.

        If `lr` and/or `optimizer_name` differ from the current values,
        hot-swap them before stepping — that's the path the UI controls
        take, so changes take effect on the next batch without requiring
        a Re-Initialize. Switching optimizer rebuilds it (loses Adam
        moments / SGD momentum, which is the only honest behavior when
        the user picks a fundamentally different update rule).
        """
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        if not images:
            raise ValueError("empty batch")

        # Optimizer change first so a combined (lr + optimizer) update
        # constructs the new optimizer at the requested lr in one shot.
        if optimizer_name is not None and optimizer_name != self.optimizer_name:
            new_lr = float(lr) if lr is not None else self.lr
            self.optimizer = make_optimizer(
                optimizer_name, self.model.parameters(), new_lr
            )
            self.optimizer_name = optimizer_name
            self.lr = new_lr
        elif lr is not None and float(lr) != self.lr:
            self.set_lr(lr)
        try:
            indices = [self.label_to_index[lab] for lab in labels]
        except KeyError as e:
            raise ValueError(f"label {e!r} not in this session's class list")

        self.model.train()
        x = _decode_batch(images, self.device)
        y = torch.tensor(indices, dtype=torch.long, device=self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # Top-1 training accuracy on this batch — measured before the step
        # so it reflects the model the user just saw predict.
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = float((preds == y).float().mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return {"loss": float(loss.item()), "step": self.step, "accuracy": accuracy}

    # ── meta ──────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# ── Checkpoint I/O ────────────────────────────────────────────────────


def save_checkpoint(session: TrainingSession, filename: str) -> str:
    CKPT_DIR.mkdir(exist_ok=True)
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    path = CKPT_DIR / filename
    # Pull from the live session attrs rather than session.hyperparameters
    # so hot-swapped values (lr, optimizer) get persisted correctly. The
    # original dict is set at init time and never updated.
    torch.save(
        {
            "state_dict": session.model.state_dict(),
            "layers": session.layers,
            "hyperparameters": {
                "lr": session.lr,
                "batch_size": session.batch_size,
                "optimizer": session.optimizer_name,
            },
            "classes": session.classes,
            "step": session.step,
        },
        path,
    )
    return path.name


def load_checkpoint(filename: str, device: torch.device) -> TrainingSession:
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    path = CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    session = TrainingSession(
        ckpt["layers"], ckpt["hyperparameters"], ckpt["classes"], device,
    )
    session.model.load_state_dict(ckpt["state_dict"])
    session.step = int(ckpt.get("step", 0))
    return session


def list_checkpoints() -> list[str]:
    if not CKPT_DIR.exists():
        return []
    return sorted(p.name for p in CKPT_DIR.glob("*.pt"))


def checkpoint_exists(filename: str) -> bool:
    if not filename:
        return False
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    return (CKPT_DIR / filename).exists()

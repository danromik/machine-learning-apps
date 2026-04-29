"""Training session for the Image Classifier.

Builds a torch model from either:
  • a user-designed sequential layer list (vocabulary: conv2d, maxpool2d,
    flatten, linear, relu, dropout, batchnorm2d); or
  • a locked preset name like 'resnet18' that bypasses the layer-list
    builder and constructs the network directly via torchvision.

A final `nn.Linear → num_classes` is appended implicitly to layer-list
architectures — the user's "Output" block represents this projection
plus the softmax head. Locked presets supply their own classifier head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from dataset import decode_png_to_array
from ml import INPUT_SHAPE


CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

MAX_LOSS_HISTORY = 2000


# ── Model construction ────────────────────────────────────────────────


def build_layered_model(layers: list[dict[str, Any]], num_classes: int) -> nn.Module:
    """Build a torch.nn.Sequential from a sequential layer list."""
    in_shape: tuple[int, ...] = tuple(INPUT_SHAPE)
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
        elif t == "batchnorm2d":
            modules.append(nn.BatchNorm2d(in_shape[0]))
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

    modules.append(nn.Linear(in_shape[0], num_classes))
    return nn.Sequential(*modules)


def build_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 from torchvision with the FC retargeted to num_classes.
    Random init (no pretrained weights) — the user is here to *train* it,
    not to use a pretrained model."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(
    layers: list[dict[str, Any]], preset: str | None, num_classes: int
) -> nn.Module:
    """Pick a model constructor. Locked preset takes priority; otherwise
    the layer list is used."""
    if preset == "resnet18":
        return build_resnet18(num_classes)
    return build_layered_model(layers, num_classes)


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


# ── Image decoding ────────────────────────────────────────────────────
#
# ImageNet normalization stats — included so layer-list architectures
# (LeNet, AlexNet, custom) train against centered inputs without the user
# having to know about it. ResNet-18 from torchvision was designed for
# these specific values, so it really matters there.

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode_image(png_b64: str) -> torch.Tensor:
    """Decode a base64 PNG → (3, H, W) float32 tensor, normalized."""
    arr = decode_png_to_array(png_b64).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(arr))


def _decode_batch(images: list[str], device: torch.device) -> torch.Tensor:
    return torch.stack([_decode_image(b) for b in images]).to(device)


# ── Training session ──────────────────────────────────────────────────


class TrainingSession:
    """Holds the model + optimizer + class-index map across requests."""

    def __init__(
        self,
        layers: list[dict[str, Any]],
        preset: str | None,
        hyperparameters: dict[str, Any],
        classes: list[str],
        device: torch.device,
        dataset_config: dict[str, Any] | None = None,
    ) -> None:
        if not classes:
            raise ValueError("classes list is empty")
        self.layers = layers
        self.preset = preset
        self.hyperparameters = dict(hyperparameters)
        self.classes = list(classes)
        self.num_classes = len(self.classes)
        self.label_to_index = {c: i for i, c in enumerate(self.classes)}
        self.device = device
        # Snapshot of the data acquisition config (augmentation flags, etc.)
        # the model was built against. Persisted in checkpoints so loading
        # restores the full pipeline alongside the weights.
        self.dataset_config = (
            dict(dataset_config) if dataset_config is not None else None
        )

        self.lr = float(hyperparameters["lr"])
        self.batch_size = int(hyperparameters["batch_size"])
        self.optimizer_name = str(hyperparameters["optimizer"])

        self.model = build_model(layers, preset, self.num_classes).to(device)
        self.optimizer = make_optimizer(
            self.optimizer_name, self.model.parameters(), self.lr
        )
        self.step = 0
        self.loss_history: list[dict[str, float]] = []
        self.val_loss_history: list[dict[str, float]] = []

    # ── inference / training ──────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images: list[str]) -> list[list[float]]:
        if not images:
            return []
        self.model.eval()
        x = _decode_batch(images, self.device)
        probs = F.softmax(self.model(x), dim=1)
        return probs.cpu().tolist()

    @torch.no_grad()
    def eval_batch(
        self, images: list[str], labels: list[str]
    ) -> dict[str, Any]:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        if not images:
            raise ValueError("empty batch")
        try:
            indices = [self.label_to_index[lab] for lab in labels]
        except KeyError as e:
            raise ValueError(f"label {e!r} not in this session's class list")
        self.model.eval()
        x = _decode_batch(images, self.device)
        y = torch.tensor(indices, dtype=torch.long, device=self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        accuracy = float((preds == y).float().mean().item())
        loss_value = float(loss.item())
        self.val_loss_history.append({"step": self.step, "loss": loss_value})
        if len(self.val_loss_history) > MAX_LOSS_HISTORY:
            self.val_loss_history = self.val_loss_history[-MAX_LOSS_HISTORY:]
        return {"loss": loss_value, "accuracy": accuracy}

    def set_lr(self, lr: float) -> None:
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
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        if not images:
            raise ValueError("empty batch")

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

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            accuracy = float((preds == y).float().mean().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        loss_value = float(loss.item())
        self.loss_history.append({"step": self.step, "loss": loss_value})
        if len(self.loss_history) > MAX_LOSS_HISTORY:
            self.loss_history = self.loss_history[-MAX_LOSS_HISTORY:]
        return {"loss": loss_value, "step": self.step, "accuracy": accuracy}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# ── Checkpoint I/O ────────────────────────────────────────────────────


def save_checkpoint(session: TrainingSession, filename: str) -> str:
    CKPT_DIR.mkdir(exist_ok=True)
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    path = CKPT_DIR / filename
    torch.save(
        {
            "state_dict": session.model.state_dict(),
            "layers": session.layers,
            "preset": session.preset,
            "hyperparameters": {
                "lr": session.lr,
                "batch_size": session.batch_size,
                "optimizer": session.optimizer_name,
            },
            "classes": session.classes,
            "step": session.step,
            "dataset_config": session.dataset_config,
            "loss_history": list(session.loss_history),
            "val_loss_history": list(session.val_loss_history),
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
        ckpt["layers"],
        ckpt.get("preset"),
        ckpt["hyperparameters"],
        ckpt["classes"],
        device,
        dataset_config=ckpt.get("dataset_config"),
    )
    session.model.load_state_dict(ckpt["state_dict"])
    session.step = int(ckpt.get("step", 0))
    session.loss_history = list(ckpt.get("loss_history") or [])
    session.val_loss_history = list(ckpt.get("val_loss_history") or [])
    return session


def list_checkpoints() -> list[dict[str, Any]]:
    if not CKPT_DIR.exists():
        return []
    files = []
    for p in sorted(CKPT_DIR.glob("*.pt"), key=lambda x: x.name):
        try:
            st = p.stat()
            files.append({
                "name": p.name,
                "size": int(st.st_size),
                "mtime": float(st.st_mtime),
            })
        except OSError:
            continue
    return files


def checkpoint_exists(filename: str) -> bool:
    if not filename:
        return False
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    return (CKPT_DIR / filename).exists()


def delete_checkpoint(filename: str) -> str:
    if not filename:
        raise FileNotFoundError("empty filename")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"invalid filename: {filename!r}")
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    path = CKPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(filename)
    path.unlink()
    return path.name

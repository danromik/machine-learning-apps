"""ML helpers used by the FastAPI server: counting, sampling, prediction,
checkpoint listing/loading, and architecture description.

Sibling-imports `models` for the model classes, device picker, and loaders.
"""

from __future__ import annotations

import base64
import io

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import (
    CKPT_DIR,
    CNN,
    DATA_DIR,
    MLP,
    build_model,
    get_loaders,
    pick_device,
)

__all__ = [
    "CKPT_DIR",
    "DATA_DIR",
    "MLP",
    "CNN",
    "build_model",
    "get_loaders",
    "pick_device",
    "list_checkpoints",
    "load_checkpoint",
    "count_params",
    "describe_architecture",
    "count_mnist",
    "sample_mnist",
    "predict_from_data_url",
    "render_canvas_preview",
]

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])


def list_checkpoints() -> list[str]:
    if not CKPT_DIR.exists():
        return []
    return sorted(p.name for p in CKPT_DIR.glob("*.pt"))


def load_checkpoint(name: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(CKPT_DIR / name, map_location=device)
    model = build_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def count_params(name: str) -> int:
    """Total parameter count for a freshly-built model of the given name."""
    return sum(p.numel() for p in build_model(name).parameters())


def describe_architecture(name: str) -> dict:
    """Layer-by-layer description of the architecture, framework-agnostic.

    The shape is intentionally generic so a future "build your own architecture"
    feature can produce the same format from a user-defined layer list.

    Each layer dict has:
      type:    one of 'input', 'linear', 'conv2d', 'maxpool2d',
               'flatten', 'activation', 'dropout', 'output' (extend freely).
      label:   short human-readable name.
      shape:   output shape after this layer (excluding batch dim).
      size:    total number of elements in that shape.
      details: type-specific metadata (kernel sizes, in/out features, etc).
    """
    if name == "mlp":
        return _describe_mlp()
    if name == "cnn":
        return _describe_cnn()
    raise ValueError(f"unknown model: {name}")


def _describe_mlp(hidden: int = 256) -> dict:
    in_dim = 28 * 28
    out_dim = 10
    layers = [
        {"type": "input", "label": "Input", "shape": [1, 28, 28], "size": in_dim, "details": {}},
        {"type": "flatten", "label": "Flatten", "shape": [in_dim], "size": in_dim, "details": {}},
        {"type": "linear", "label": "Linear", "shape": [hidden], "size": hidden,
         "details": {"in_features": in_dim, "out_features": hidden}},
        {"type": "activation", "label": "ReLU", "shape": [hidden], "size": hidden,
         "details": {"fn": "relu"}},
        {"type": "linear", "label": "Linear", "shape": [hidden], "size": hidden,
         "details": {"in_features": hidden, "out_features": hidden}},
        {"type": "activation", "label": "ReLU", "shape": [hidden], "size": hidden,
         "details": {"fn": "relu"}},
        {"type": "linear", "label": "Linear", "shape": [out_dim], "size": out_dim,
         "details": {"in_features": hidden, "out_features": out_dim}},
        {"type": "output", "label": "Output", "shape": [out_dim], "size": out_dim, "details": {}},
    ]
    return {"name": "MLP", "layers": layers}


def _describe_cnn() -> dict:
    layers = [
        {"type": "input", "label": "Input", "shape": [1, 28, 28], "size": 1 * 28 * 28, "details": {}},
        {"type": "conv2d", "label": "Conv2D", "shape": [32, 28, 28], "size": 32 * 28 * 28,
         "details": {"in_channels": 1, "out_channels": 32, "kernel": 3, "padding": 1}},
        {"type": "activation", "label": "ReLU", "shape": [32, 28, 28], "size": 32 * 28 * 28,
         "details": {"fn": "relu"}},
        {"type": "maxpool2d", "label": "MaxPool", "shape": [32, 14, 14], "size": 32 * 14 * 14,
         "details": {"kernel": 2}},
        {"type": "conv2d", "label": "Conv2D", "shape": [64, 14, 14], "size": 64 * 14 * 14,
         "details": {"in_channels": 32, "out_channels": 64, "kernel": 3, "padding": 1}},
        {"type": "activation", "label": "ReLU", "shape": [64, 14, 14], "size": 64 * 14 * 14,
         "details": {"fn": "relu"}},
        {"type": "maxpool2d", "label": "MaxPool", "shape": [64, 7, 7], "size": 64 * 7 * 7,
         "details": {"kernel": 2}},
        {"type": "flatten", "label": "Flatten", "shape": [64 * 7 * 7], "size": 64 * 7 * 7, "details": {}},
        {"type": "linear", "label": "Linear", "shape": [128], "size": 128,
         "details": {"in_features": 64 * 7 * 7, "out_features": 128}},
        {"type": "activation", "label": "ReLU", "shape": [128], "size": 128, "details": {"fn": "relu"}},
        {"type": "dropout", "label": "Dropout", "shape": [128], "size": 128,
         "details": {"p": 0.25}},
        {"type": "linear", "label": "Linear", "shape": [10], "size": 10,
         "details": {"in_features": 128, "out_features": 10}},
        {"type": "output", "label": "Output", "shape": [10], "size": 10, "details": {}},
    ]
    return {"name": "CNN", "layers": layers}


# ── MNIST sampling & counting (cached so repeated requests are cheap) ────

_ds_cache: dict[bool, object] = {}
_index_cache: dict[tuple, list[tuple[int, int]]] = {}


def _get_ds(is_train: bool):
    ds = _ds_cache.get(is_train)
    if ds is None:
        from torchvision import datasets
        ds = datasets.MNIST(DATA_DIR, train=is_train, download=True, transform=None)
        _ds_cache[is_train] = ds
    return ds


def _get_index(split: str, class_filter: int | None, order: str) -> list[tuple[int, int]]:
    """Cached (idx, label) pairs after applying class filter and ordering."""
    key = (split, class_filter, order)
    cached = _index_cache.get(key)
    if cached is not None:
        return cached
    ds = _get_ds(split == "train")
    targets = ds.targets.tolist()
    pairs = list(enumerate(targets))
    if class_filter is not None:
        pairs = [p for p in pairs if p[1] == class_filter]
    if order == "digit":
        pairs.sort(key=lambda p: (p[1], p[0]))
    _index_cache[key] = pairs
    return pairs


def count_mnist(class_filter: int | None) -> dict[str, int]:
    """Return {'train': n, 'test': n} total items, optionally restricted to one class."""
    train_ds = _get_ds(True)
    test_ds = _get_ds(False)
    if class_filter is None:
        return {"train": len(train_ds), "test": len(test_ds)}
    return {
        "train": int((train_ds.targets == class_filter).sum()),
        "test": int((test_ds.targets == class_filter).sum()),
    }


def sample_mnist(
    split: str,
    class_filter: int | None,
    n: int,
    order: str = "default",
    offset: int = 0,
) -> list[dict]:
    """Return up to n MNIST examples as {png_b64, label} dicts, starting at `offset`.

    order="default" → raw dataset order (as packaged on disk).
    order="digit"   → sorted by label (0,0,…,1,1,…).
    """
    is_train = split == "train"
    ds = _get_ds(is_train)
    pairs = _get_index(split, class_filter, order)

    if offset < 0:
        offset = 0
    window = pairs[offset : offset + n]

    out: list[dict] = []
    for idx, label in window:
        img, _ = ds[idx]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({
            "png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "label": int(label),
        })
    return out


@torch.no_grad()
def predict_from_data_url(data_url: str, model: torch.nn.Module, device: torch.device) -> dict:
    """Given a 'data:image/png;base64,...' URL from a canvas, return top-k probs."""
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    # Canvas is white strokes on black — already matches MNIST orientation.
    img = img.resize((28, 28), Image.LANCZOS)
    x = _normalize(img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
    return {"probs": probs, "pred": int(torch.tensor(probs).argmax())}


def render_canvas_preview(data_url: str) -> str:
    """Downsample the canvas to 28x28 and return a 4x-nearest upscale as base64 PNG."""
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L").resize((28, 28), Image.LANCZOS)
    big = img.resize((112, 112), Image.NEAREST)
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

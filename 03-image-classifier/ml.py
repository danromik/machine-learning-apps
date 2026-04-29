"""Static metadata for the Image Classifier.

Exposes:
  - IMAGENETTE_CLASSES: the 10-class Imagenette catalog (wnid + label).
  - INPUT_SIZE / NUM_CHANNELS: tensor shape every model in this app sees.
  - ARCHITECTURE_PRESETS: three named architectures the user can drop onto
    the canvas — `lenet5`, `alexnet`, and `resnet18`. The first two are
    sequential layer lists (editable after applying); `resnet18` is a
    locked preset built directly in `training.py` (residual connections
    don't fit the linear drag-and-drop layer model).

Imagenette is fast.ai's 10-class subset of ImageNet — a teaching-friendly
stand-in for the full ILSVRC dataset that AlexNet was trained on. Same
kind of natural images, but trains in minutes instead of days.
"""
from __future__ import annotations

from typing import Any


# ── Image tensor shape ─────────────────────────────────────────────────
#
# Every model in this app sees 3×96×96 RGB inputs. 96 is a compromise:
# big enough to preserve recognizable structure in natural photos, small
# enough that AlexNet on MPS trains at interactive speed. Imagenette
# images are downscaled to this size at load time.

INPUT_SIZE = 96
NUM_CHANNELS = 3
INPUT_SHAPE = (NUM_CHANNELS, INPUT_SIZE, INPUT_SIZE)


# ── Imagenette catalog ─────────────────────────────────────────────────

IMAGENETTE_CLASSES: list[dict[str, str]] = [
    {"wnid": "n01440764", "label": "tench"},
    {"wnid": "n02102040", "label": "English springer"},
    {"wnid": "n02979186", "label": "cassette player"},
    {"wnid": "n03000684", "label": "chain saw"},
    {"wnid": "n03028079", "label": "church"},
    {"wnid": "n03394916", "label": "French horn"},
    {"wnid": "n03417042", "label": "garbage truck"},
    {"wnid": "n03425413", "label": "gas pump"},
    {"wnid": "n03445777", "label": "golf ball"},
    {"wnid": "n03888257", "label": "parachute"},
]

IMAGENETTE_LABELS = [c["label"] for c in IMAGENETTE_CLASSES]
IMAGENETTE_WNIDS = [c["wnid"] for c in IMAGENETTE_CLASSES]


def list_classes() -> list[dict[str, Any]]:
    """Class table the model is trained against. Index is the integer
    label the model uses internally."""
    return [
        {"index": i, "wnid": c["wnid"], "label": c["label"]}
        for i, c in enumerate(IMAGENETTE_CLASSES)
    ]


# ── Dataset source ─────────────────────────────────────────────────────
#
# 160-px variant: ~88 MB, the smallest dimension is resized to 160 — plenty
# of resolution for our 96×96 inputs.

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_ARCHIVE = "imagenette2-160.tgz"
IMAGENETTE_DIR = "imagenette2-160"


# ── Architecture presets ───────────────────────────────────────────────
#
# Each preset is one of:
#   • a sequential layer list (`layers`) the user can edit after dropping
#     it onto the canvas (LeNet-5, AlexNet);
#   • a `locked` flag (`locked: True`) meaning the architecture is built
#     in Python by name (ResNet-18). The diagram shows a single
#     "ResNet-18" placeholder block in this mode.

# LeNet-5 (LeCun et al., 1998). Two conv→pool blocks feeding a small
# dense head — original target was 32×32 grayscale digits; we keep the
# topology and channel counts and let it operate on 96×96 RGB.
LENET5_LAYERS: list[dict[str, Any]] = [
    {"type": "conv2d", "params": {"out_channels": 6, "kernel": 5, "padding": 0, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel": 2, "stride": 2}},
    {"type": "conv2d", "params": {"out_channels": 16, "kernel": 5, "padding": 0, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel": 2, "stride": 2}},
    {"type": "flatten", "params": {}},
    {"type": "linear", "params": {"out_features": 120}},
    {"type": "relu", "params": {}},
    {"type": "linear", "params": {"out_features": 84}},
    {"type": "relu", "params": {}},
]

# AlexNet (Krizhevsky, Sutskever, Hinton 2012), adapted for 96×96 RGB.
# 5-conv / 3-FC topology preserved; FC width shrunk from 4096 to 1024
# since 4096 on 96×96 features pushes parameters past 100M for a
# teaching app.
ALEXNET_LAYERS: list[dict[str, Any]] = [
    {"type": "conv2d", "params": {"out_channels": 64, "kernel": 11, "padding": 2, "stride": 4}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel": 3, "stride": 2}},
    {"type": "conv2d", "params": {"out_channels": 192, "kernel": 5, "padding": 2, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel": 3, "stride": 2}},
    {"type": "conv2d", "params": {"out_channels": 384, "kernel": 3, "padding": 1, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "conv2d", "params": {"out_channels": 256, "kernel": 3, "padding": 1, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "conv2d", "params": {"out_channels": 256, "kernel": 3, "padding": 1, "stride": 1}},
    {"type": "relu", "params": {}},
    {"type": "maxpool2d", "params": {"kernel": 3, "stride": 2}},
    {"type": "flatten", "params": {}},
    {"type": "dropout", "params": {"p": 0.5}},
    {"type": "linear", "params": {"out_features": 1024}},
    {"type": "relu", "params": {}},
    {"type": "dropout", "params": {"p": 0.5}},
    {"type": "linear", "params": {"out_features": 1024}},
    {"type": "relu", "params": {}},
]


ARCHITECTURE_PRESETS: dict[str, dict[str, Any]] = {
    "lenet5": {
        "name": "lenet5",
        "label": "LeNet-5",
        "year": 1998,
        "tagline": "The classic pre-AlexNet CNN — small, fast, surprisingly capable.",
        "description": (
            "LeCun et al.'s 1998 architecture for handwritten-digit recognition. "
            "Two conv→pool blocks feed a small dense head. ReLUs replace the "
            "original sigmoids since modern tooling makes them strictly better."
        ),
        "layers": LENET5_LAYERS,
        "locked": False,
        "hyperparameters": {"lr": 0.001, "batch_size": 64, "optimizer": "adam"},
    },
    "alexnet": {
        "name": "alexnet",
        "label": "AlexNet",
        "year": 2012,
        "tagline": "The architecture that kicked off the deep-learning revolution at ILSVRC 2012.",
        "description": (
            "Krizhevsky, Sutskever, and Hinton's 5-conv / 3-FC design with ReLU + dropout. "
            "Adapted here for 96×96 input and a 10-class head; FC width shrunk from 4096 "
            "to 1024 to keep parameter count tractable on a single GPU."
        ),
        "layers": ALEXNET_LAYERS,
        "locked": False,
        "hyperparameters": {"lr": 0.001, "batch_size": 64, "optimizer": "adam"},
    },
    "resnet18": {
        "name": "resnet18",
        "label": "ResNet-18",
        "year": 2015,
        "tagline": "He et al.'s residual networks — skip connections that finally let networks go deep.",
        "description": (
            "ResNet-18 from torchvision, with the final FC retargeted to the 10 "
            "Imagenette classes. Locked preset: residual connections don't fit the "
            "linear drag-and-drop layer model, so the diagram shows a single "
            "placeholder block instead of the per-layer graph."
        ),
        "layers": [],
        "locked": True,
        "hyperparameters": {"lr": 0.001, "batch_size": 32, "optimizer": "adam"},
    },
}

PRESET_ORDER = ["lenet5", "alexnet", "resnet18"]


def list_architecture_presets() -> list[dict[str, Any]]:
    return [ARCHITECTURE_PRESETS[name] for name in PRESET_ORDER]


def get_architecture_preset(name: str) -> dict[str, Any]:
    if name not in ARCHITECTURE_PRESETS:
        raise ValueError(f"unknown architecture preset: {name}")
    return ARCHITECTURE_PRESETS[name]

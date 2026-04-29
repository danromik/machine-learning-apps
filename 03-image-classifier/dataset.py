"""Imagenette dataset acquisition and batch loading.

Pipeline:
  1. download_imagenette()  →  pulls fast.ai's 160-px tarball into data/,
                               extracts it, returns dataset_status().
  2. dataset_status()       →  reports whether the dataset is on disk
                               and counts train/val images per class.
  3. ImagenetteIndex        →  in-memory list of (image_path, class_idx)
                               for one split; built lazily on first use.
  4. sample_batch()         →  random batch of size N (with augmentation
                               if split=='train') as base64 PNGs + labels.

The base64-PNG round-trip exists so the frontend can show actual training
batches in the UI without a second request — same shape as 02-math-symbols
returns from /api/synthesis/sample. It's not the cheapest path; for headless
CLI training you'd want to skip the encode and feed tensors directly. We
cap that cost here by resizing to 96×96 before encoding so PNGs stay tiny.
"""
from __future__ import annotations

import base64
import io
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from ml import (
    IMAGENETTE_ARCHIVE,
    IMAGENETTE_CLASSES,
    IMAGENETTE_DIR,
    IMAGENETTE_LABELS,
    IMAGENETTE_URL,
    IMAGENETTE_WNIDS,
    INPUT_SIZE,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE_PATH = DATA_DIR / IMAGENETTE_ARCHIVE
EXTRACT_PATH = DATA_DIR / IMAGENETTE_DIR


# ── Status / Download ──────────────────────────────────────────────────


def _list_split_files(split: str) -> dict[str, list[Path]]:
    """Map wnid → sorted list of image paths in <extract>/<split>/<wnid>/."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    root = EXTRACT_PATH / split
    out: dict[str, list[Path]] = {}
    for wnid in IMAGENETTE_WNIDS:
        d = root / wnid
        if not d.is_dir():
            out[wnid] = []
            continue
        out[wnid] = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    return out


def dataset_status() -> dict[str, Any]:
    """Snapshot of what's on disk: tarball present?, extracted?, per-class
    train/val counts. Returned to the frontend so the Data Acquisition tab
    can show download state at a glance."""
    archive_present = ARCHIVE_PATH.exists()
    extracted = EXTRACT_PATH.is_dir() and (EXTRACT_PATH / "train").is_dir()

    train_files: dict[str, list[Path]] = {}
    val_files: dict[str, list[Path]] = {}
    if extracted:
        train_files = _list_split_files("train")
        val_files = _list_split_files("val")

    per_class: list[dict[str, Any]] = []
    for c in IMAGENETTE_CLASSES:
        wnid = c["wnid"]
        per_class.append(
            {
                "wnid": wnid,
                "label": c["label"],
                "train": len(train_files.get(wnid, [])),
                "val": len(val_files.get(wnid, [])),
            }
        )

    archive_size = ARCHIVE_PATH.stat().st_size if archive_present else 0
    extract_size = _du(EXTRACT_PATH) if extracted else 0

    return {
        "archive_present": archive_present,
        "archive_size": archive_size,
        "extracted": extracted,
        "extract_size": extract_size,
        "data_dir": str(DATA_DIR),
        "url": IMAGENETTE_URL,
        "num_train": sum(c["train"] for c in per_class),
        "num_val": sum(c["val"] for c in per_class),
        "per_class": per_class,
    }


def _du(path: Path) -> int:
    """Recursive size in bytes."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                continue
    return total


def download_imagenette(progress_cb=None) -> dict[str, Any]:
    """Download the Imagenette tarball if missing, then extract if missing.
    Idempotent: returns the post-action status without re-downloading or
    re-extracting if both steps are already done.

    `progress_cb`, if provided, is called periodically as
        progress_cb(stage, downloaded_bytes, total_bytes)
    where `stage` ∈ {'download', 'extract'} and totals may be 0 if unknown.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if not ARCHIVE_PATH.exists():
        tmp = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".part")
        try:
            req = urllib.request.Request(
                IMAGENETTE_URL,
                headers={"User-Agent": "ml-image-classifier/0.1"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                downloaded = 0
                last_report = 0
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and downloaded - last_report > 1024 * 1024:
                        progress_cb("download", downloaded, total)
                        last_report = downloaded
                if progress_cb:
                    progress_cb("download", downloaded, total or downloaded)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
        tmp.rename(ARCHIVE_PATH)

    if not EXTRACT_PATH.is_dir() or not (EXTRACT_PATH / "train").is_dir():
        if progress_cb:
            progress_cb("extract", 0, 0)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
            tf.extractall(DATA_DIR)
        if progress_cb:
            progress_cb("extract", 1, 1)

    return dataset_status()


# ── Index + batch loader ───────────────────────────────────────────────


@dataclass
class ImagenetteIndex:
    """In-memory list of (image_path, class_idx) pairs for one split."""
    split: str
    items: list[tuple[Path, int]]

    @classmethod
    def build(cls, split: str) -> "ImagenetteIndex":
        files = _list_split_files(split)
        wnid_to_idx = {w: i for i, w in enumerate(IMAGENETTE_WNIDS)}
        items: list[tuple[Path, int]] = []
        for wnid, paths in files.items():
            idx = wnid_to_idx[wnid]
            for p in paths:
                items.append((p, idx))
        return cls(split=split, items=items)


# Cache split indices so consecutive batches don't re-walk the directory
# tree. Invalidated by reset_indices() — called after a download so the
# fresh files get picked up.
_index_cache: dict[str, ImagenetteIndex] = {}


def get_index(split: str) -> ImagenetteIndex:
    if split not in _index_cache:
        _index_cache[split] = ImagenetteIndex.build(split)
    return _index_cache[split]


def reset_indices() -> None:
    _index_cache.clear()


# ── Augmentation + image preprocessing ─────────────────────────────────
#
# Two pipelines, mirroring standard ImageNet practice:
#   • train: resize-shorter-side → random crop → random horizontal flip.
#   • val:   resize-shorter-side → center crop. Deterministic.
# Color-jitter and random rotation are exposed as toggles in the
# Data Acquisition tab so the user can experiment with them.


def _resize_shorter(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    scale = target / min(w, h)
    new_w = max(target, int(round(w * scale)))
    new_h = max(target, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _random_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    w, h = img.size
    if w == size and h == size:
        return img
    x = rng.randint(0, w - size)
    y = rng.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    x = (w - size) // 2
    y = (h - size) // 2
    return img.crop((x, y, x + size, y + size))


def _color_jitter(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Multiplicative brightness/contrast/saturation jitter. `strength` ∈
    [0, 1] sets the half-range — at 0.4 we draw factors uniformly from
    [0.6, 1.4], standard ImageNet practice."""
    from PIL import ImageEnhance

    s = max(0.0, min(1.0, strength))

    def factor() -> float:
        return 1.0 + rng.uniform(-s, s)

    img = ImageEnhance.Brightness(img).enhance(factor())
    img = ImageEnhance.Contrast(img).enhance(factor())
    img = ImageEnhance.Color(img).enhance(factor())
    return img


@dataclass
class BatchRequest:
    split: str  # 'train' | 'val'
    count: int
    seed: int = 42
    flip: bool = True
    jitter: float = 0.0  # 0 disables; 0.4 is a good default
    random_crop: bool = True


def sample_batch(req: BatchRequest) -> list[dict[str, Any]]:
    """Return a list of `{ png_b64, label, label_index, source }` dicts.

    Train: resize-shorter → random_crop (or center) → optional hflip + jitter.
    Val:   resize-shorter → center_crop. Deterministic for a given seed so
    the validation curve doesn't bounce on which images got drawn.
    """
    idx = get_index(req.split)
    if not idx.items:
        return []

    rng = random.Random(req.seed)
    is_train = req.split == "train"
    target = INPUT_SIZE
    # Shorter-side target slightly larger than the crop so random_crop has
    # somewhere to move. 110/96 ≈ 1.15x — small enough that train images
    # don't lose much resolution.
    resize_target = int(round(target * 1.15))

    out: list[dict[str, Any]] = []
    for _ in range(req.count):
        path, label_idx = rng.choice(idx.items)
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            continue

        img = _resize_shorter(img, resize_target)

        if is_train and req.random_crop:
            img = _random_crop(img, target, rng)
        else:
            img = _center_crop(img, target)

        if is_train:
            if req.flip and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if req.jitter > 0:
                img = _color_jitter(img, req.jitter, rng)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        out.append(
            {
                "png_b64": png_b64,
                "label": IMAGENETTE_LABELS[label_idx],
                "label_index": label_idx,
                "source": path.name,
            }
        )

    return out


def sample_batch_iter(req: BatchRequest) -> Iterator[dict[str, Any]]:
    """Streaming variant — same as sample_batch but yields one at a time
    so the frontend's preview modal can render samples as they arrive."""
    idx = get_index(req.split)
    if not idx.items:
        return

    rng = random.Random(req.seed)
    is_train = req.split == "train"
    target = INPUT_SIZE
    resize_target = int(round(target * 1.15))

    for _ in range(req.count):
        path, label_idx = rng.choice(idx.items)
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            continue

        img = _resize_shorter(img, resize_target)

        if is_train and req.random_crop:
            img = _random_crop(img, target, rng)
        else:
            img = _center_crop(img, target)

        if is_train:
            if req.flip and rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if req.jitter > 0:
                img = _color_jitter(img, req.jitter, rng)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        yield {
            "png_b64": png_b64,
            "label": IMAGENETTE_LABELS[label_idx],
            "label_index": label_idx,
            "source": path.name,
        }


# ── Inference helpers ──────────────────────────────────────────────────


def encode_image_for_inference(img_bytes: bytes) -> str:
    """Decode a user-uploaded image, run the val-pipeline preprocessing
    (resize + center crop), and return a 96×96 base64 PNG. Same exact
    pipeline the model sees during evaluation, so what the user uploads
    really is what the model classifies."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = _resize_shorter(img, int(round(INPUT_SIZE * 1.15)))
    img = _center_crop(img, INPUT_SIZE)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_png_to_array(png_b64: str) -> np.ndarray:
    """Decode base64 PNG → (H, W, 3) uint8 ndarray. Used by training.py
    to build batched tensors."""
    raw = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)

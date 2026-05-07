"""On-the-fly glyph synthesis for the Agentic Symbol Trainer.

Pipeline per sample:
    pick label  →  pick font  →  render glyph centered  →  augment  →  PNG bytes

Internal representation: 8-bit grayscale, black-on-white (printed-text OCR
convention). The frontend inverts in dark mode via CSS so the displayed
content matches the active theme.

Rendering: render at 2× the final canvas size, then downsample with LANCZOS
for clean antialiasing. Glyph centered using the font's tight bounding box;
glyph height is targeted to ~70% of the canvas so different aspect ratios
(narrow ':', wide '∑') stay roughly balanced visually.

Glyph coverage: fontTools' cmap tells us whether a font actually contains
the requested codepoint. PIL falls back to the `.notdef` placeholder (an
empty box) when the glyph is missing — we render it anyway, but tag the
sample as `missing_glyph: True` so the UI can surface it. The training
loop, when it lands, should skip missing-glyph (label, font) pairs.
"""
from __future__ import annotations

import base64
import io
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTCollection, TTFont

from ml import SYMBOL_CATEGORIES, _installed_families, font_path


# ── Tunables ───────────────────────────────────────────────────────────

CANVAS_SIZE = 64                  # final image side, in pixels
RENDER_SIZE = 128                 # 2× supersampling for clean antialiasing
GLYPH_TARGET_HEIGHT_FRAC = 0.70   # glyph fills ~70% of the canvas height
SHEAR_RANGE_DEG = 15.0            # ± shear angle for the skew augmentation
NOISE_MAX_SIGMA = 0.40            # σ at max_level=100 (in normalized [0,1])


@dataclass
class SynthesisRequest:
    categories: list[str]
    training_fonts: list[str]
    validation_fonts: list[str]
    augmentation: dict
    split: str  # 'train' or 'val'
    count: int = 500
    seed: int = 42


# ── Glyph coverage check ───────────────────────────────────────────────


@lru_cache(maxsize=128)
def _font_codepoints(path: str) -> frozenset[int]:
    """Set of Unicode codepoints supported by the font (from its cmap)."""
    try:
        if path.lower().endswith(".ttc"):
            coll = TTCollection(path)
            font = coll.fonts[0]
        else:
            font = TTFont(path)
        cmap = font.getBestCmap()
        return frozenset(cmap.keys()) if cmap else frozenset()
    except Exception:
        return frozenset()


def _has_glyph(char: str, path: str) -> bool:
    if not char:
        return False
    return ord(char) in _font_codepoints(path)


# ── Rendering ──────────────────────────────────────────────────────────


@lru_cache(maxsize=256)
def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=0)


def _font_size_for_target_height(path: str, char: str, target_h: int) -> int:
    """Pick a font size such that `char` renders ≈ target_h pixels tall.

    Uses a single measurement at size 100 and linearly extrapolates. Glyph
    metrics scale linearly with font size, so this is exact (modulo tiny
    rounding).
    """
    f = _load_font(path, 100)
    bbox = f.getbbox(char)
    actual_h = bbox[3] - bbox[1]
    if actual_h <= 0:
        return 100
    return max(8, int(round(100 * target_h / actual_h)))


def render_glyph(char: str, path: str) -> Image.Image:
    """Render `char` from the font at `path` onto a CANVAS_SIZE×CANVAS_SIZE
    grayscale image, black-on-white, centered using the glyph's tight bbox."""
    target_h = int(RENDER_SIZE * GLYPH_TARGET_HEIGHT_FRAC)
    size = _font_size_for_target_height(path, char, target_h)
    font = _load_font(path, size)

    img = Image.new("L", (RENDER_SIZE, RENDER_SIZE), 255)
    draw = ImageDraw.Draw(img)

    bbox = font.getbbox(char)
    glyph_w = bbox[2] - bbox[0]
    glyph_h = bbox[3] - bbox[1]
    x = (RENDER_SIZE - glyph_w) // 2 - bbox[0]
    y = (RENDER_SIZE - glyph_h) // 2 - bbox[1]
    draw.text((x, y), char, fill=0, font=font)

    return img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)


# ── Augmentation ───────────────────────────────────────────────────────


def _apply_skew(img: Image.Image, rng: random.Random) -> Image.Image:
    """Random horizontal shear in ± SHEAR_RANGE_DEG, centered on canvas."""
    angle = rng.uniform(-SHEAR_RANGE_DEG, SHEAR_RANGE_DEG)
    shear = float(np.tan(np.radians(angle)))
    w, h = img.size
    # PIL's affine is the inverse map (output → input), so x_in = x_out + shear*(y_out - h/2).
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, shear, -shear * h / 2, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=255,
    )


def _apply_noise(img: Image.Image, max_level: int, rng: random.Random) -> Image.Image:
    """Add Gaussian noise; max_level (0–100) maps linearly to σ ∈ [0, NOISE_MAX_SIGMA]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    sigma = (max_level / 100.0) * NOISE_MAX_SIGMA
    seed = rng.randrange(2**32)
    noise = np.random.default_rng(seed).normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def _apply_augmentation(
    img: Image.Image, aug: dict, rng: random.Random
) -> Image.Image:
    if aug.get("skew", {}).get("enabled"):
        img = _apply_skew(img, rng)
    noise = aug.get("noise", {})
    if noise.get("enabled") and noise.get("max_level", 0) > 0:
        img = _apply_noise(img, int(noise["max_level"]), rng)
    return img


# ── Synthesis driver ───────────────────────────────────────────────────


def _collect_symbols(category_ids: list[str]) -> list[str]:
    """Flatten the selected categories' symbols into a single list."""
    symbols: list[str] = []
    for cid in category_ids:
        cat = SYMBOL_CATEGORIES.get(cid)
        if cat is None:
            continue
        symbols.extend(cat["symbols"])
    return symbols


def synthesize_iter(req: SynthesisRequest) -> Iterator[dict[str, Any]]:
    """Yield synthesized samples one at a time as
    `{ png_b64, label, font, missing_glyph }` dicts.

    Skips (char, font) pairs whose font doesn't actually contain the
    glyph — those would render as the .notdef placeholder (an empty box),
    so training on them would teach the model that "blank box → α" (etc.),
    which poisons the labels. We pre-compute, per char, the subset of the
    requested fonts that supports it and sample only from those pairs.
    Chars that no requested font supports are silently dropped.
    """
    rng = random.Random(req.seed)

    symbols = _collect_symbols(req.categories)
    if not symbols:
        return

    fonts_in = req.training_fonts if req.split == "train" else req.validation_fonts
    installed = _installed_families()
    fonts = [f for f in fonts_in if f in installed]
    if not fonts:
        return

    # Per-char list of fonts that actually contain the glyph. Computed
    # once up front; _has_glyph is cached so this is a fast pass.
    supported_by_char: dict[str, list[str]] = {}
    for char in set(symbols):
        ok = []
        for family in fonts:
            path = font_path(family)
            if path is not None and _has_glyph(char, path):
                ok.append(family)
        if ok:
            supported_by_char[char] = ok

    valid_chars = [c for c in symbols if c in supported_by_char]
    if not valid_chars:
        return

    for _ in range(req.count):
        char = rng.choice(valid_chars)
        family = rng.choice(supported_by_char[char])
        path = font_path(family)
        if path is None:
            continue

        img = render_glyph(char, path)
        img = _apply_augmentation(img, req.augmentation, rng)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        yield {
            "png_b64": png_b64,
            "label": char,
            "font": family,
            # Always False after the pre-filter, but kept in the response
            # so the frontend's display path stays uniform.
            "missing_glyph": False,
        }

"""Helpers for the Math Symbol Trainer.

Currently exposes the data-synthesis configuration surface:
  - SYMBOL_CATEGORIES: master list of glyphs we can train on, grouped by category
  - list_categories(): categories in display order
  - discover_fonts(): scan installed fonts on macOS, deduped by family
  - get_preset(name): return a SynthesisConfig dict for a named preset

Each glyph is assigned to exactly one category to avoid ambiguous training
labels. ASCII chars (+, -, =, <, >, *, /) live in their ASCII categories;
the math category contains only Unicode-only operators (−, ×, ÷, ≠, ≤, …).

Latin/Greek visual collisions (o/ο, A/Α, B/Β, …) are intentional: enabling
both categories simultaneously gives the model two labels for one glyph,
which is the pedagogical setup for a confusion-matrix lesson.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import ImageFont


# ── Symbol categories ──────────────────────────────────────────────────

SYMBOL_CATEGORIES: dict[str, dict[str, Any]] = {
    "digits": {
        "label": "Digits",
        "symbols": list("0123456789"),
    },
    "lower_roman": {
        "label": "Lowercase Roman",
        "symbols": list("abcdefghijklmnopqrstuvwxyz"),
    },
    "upper_roman": {
        "label": "Uppercase Roman",
        "symbols": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    },
    "punctuation": {
        "label": "Punctuation",
        # . , ; : ! ? ' " ` ( ) [ ] { } -
        "symbols": list(".,;:!?'\"`()[]{}-"),
    },
    "other_ascii": {
        "label": "Other ASCII",
        # @ # $ % ^ & * + = < > / \ | ~ _
        "symbols": list("@#$%^&*+=<>/\\|~_"),
    },
    "lower_greek": {
        "label": "Lowercase Greek",
        "symbols": list("αβγδεζηθικλμνξοπρστυφχψω"),
    },
    "upper_greek": {
        "label": "Uppercase Greek",
        "symbols": list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"),
    },
    "math": {
        "label": "Math symbols",
        # Unicode-only math; ASCII duplicates (+, -, *, /, =, <, >) live in
        # punctuation/other_ascii to keep labels disjoint.
        "symbols": [
            # arithmetic
            "−", "×", "÷", "±", "∓",
            # equivalence / proportion
            "≠", "≈", "≡", "≅", "≃", "∝",
            # comparison
            "≤", "≥", "≪", "≫",
            # operators / special
            "∞", "∂", "∇", "∫", "∮", "∑", "∏", "√", "∛",
            # set
            "∈", "∉", "∋", "⊂", "⊃", "⊆", "⊇", "∪", "∩", "∅",
            # logic / quantifiers
            "∀", "∃", "∄", "¬", "∧", "∨",
            # circled operators
            "⊕", "⊗", "⊙",
            # arrows
            "⇒", "⇐", "⇔", "→", "←", "↔", "↑", "↓",
            # measurement
            "°", "′", "″",
            # blackboard / script constants
            "ℵ", "ℝ", "ℂ", "ℕ", "ℤ", "ℚ", "ℏ", "ℓ",
            # geometric
            "⊥", "∥",
            # therefore / because
            "∴", "∵",
            # delimiters
            "⌊", "⌋", "⌈", "⌉", "⟨", "⟩",
            # dot operators
            "·", "•",
        ],
    },
}

CATEGORY_ORDER = [
    "digits", "lower_roman", "upper_roman", "punctuation",
    "other_ascii", "lower_greek", "upper_greek", "math",
]


def list_categories() -> list[dict[str, Any]]:
    """Return the categories in display order, with their symbols and counts."""
    return [
        {
            "id": cid,
            "label": SYMBOL_CATEGORIES[cid]["label"],
            "symbols": SYMBOL_CATEGORIES[cid]["symbols"],
            "count": len(SYMBOL_CATEGORIES[cid]["symbols"]),
        }
        for cid in CATEGORY_ORDER
    ]


# ── Curated math/science fonts ─────────────────────────────────────────

# Mathematical and scientific texts use a narrow vocabulary of fonts —
# Computer Modern (the TeX default) and Times-style serifs cover ~99% of
# math papers. We expose a curated list of ~12 fonts known for STEM use,
# and check at runtime which are actually installed.
#
# Order also serves as a priority for preset selection.
CURATED_FONTS: list[dict[str, str]] = [
    {
        "family": "Latin Modern Roman",
        "url": "https://www.gust.org.pl/projects/e-foundry/latin-modern",
        "install": "brew install --cask font-latin-modern-roman",
        "note": "Computer Modern, the TeX default — used in ~90% of math papers",
    },
    {
        "family": "Times New Roman",
        "url": "https://learn.microsoft.com/typography/font-list/times-new-roman",
        "install": "Comes with macOS / Microsoft Office",
        "note": "Standard in scientific journals (Nature, Science, Elsevier)",
    },
    {
        "family": "TeX Gyre Termes",
        "url": "https://www.gust.org.pl/projects/e-foundry/tex-gyre",
        "install": "brew install --cask font-tex-gyre-termes",
        "note": "Free Times-like clone, full math support",
    },
    {
        "family": "TeX Gyre Pagella",
        "url": "https://www.gust.org.pl/projects/e-foundry/tex-gyre",
        "install": "brew install --cask font-tex-gyre-pagella",
        "note": "Free Palatino-like clone",
    },
    {
        "family": "TeX Gyre Schola",
        "url": "https://www.gust.org.pl/projects/e-foundry/tex-gyre",
        "install": "brew install --cask font-tex-gyre-schola",
        "note": "Free Century Schoolbook-like clone",
    },
    {
        "family": "TeX Gyre Bonum",
        "url": "https://www.gust.org.pl/projects/e-foundry/tex-gyre",
        "install": "brew install --cask font-tex-gyre-bonum",
        "note": "Free Bookman-like clone",
    },
    {
        "family": "STIX Two Text",
        "url": "https://www.stixfonts.org/",
        "install": "brew install --cask font-stix-two-text",
        "note": "Designed for STEM publishing (AMS, AIP, APS journals)",
    },
    {
        "family": "Latin Modern Sans",
        "url": "https://www.gust.org.pl/projects/e-foundry/latin-modern",
        "install": "brew install --cask font-latin-modern-roman",
        "note": "Sans-serif companion to Computer Modern",
    },
    {
        "family": "Latin Modern Mono",
        "url": "https://www.gust.org.pl/projects/e-foundry/latin-modern",
        "install": "brew install --cask font-latin-modern-roman",
        "note": "Monospace companion to Computer Modern",
    },
    {
        "family": "Palatino",
        "url": "https://en.wikipedia.org/wiki/Palatino",
        "install": "Comes with macOS",
        "note": "Common in mathematical typography (Pagella in TeX)",
    },
    {
        "family": "Georgia",
        "url": "https://learn.microsoft.com/typography/font-list/georgia",
        "install": "Comes with macOS / Microsoft Office",
        "note": "Screen-friendly serif used in online math (e.g. Wikipedia)",
    },
    {
        "family": "Helvetica",
        "url": "https://en.wikipedia.org/wiki/Helvetica",
        "install": "Comes with macOS",
        "note": "Sans-serif for figure captions and titles",
    },
]


# macOS font search paths — only used to determine which curated families
# are actually installed.
_FONT_DIRS = [
    Path("/System/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path.home() / "Library" / "Fonts",
]


@lru_cache(maxsize=1)
def _family_to_path() -> dict[str, str]:
    """Map font family name → file path. Prefers the 'Regular' style file
    when a family ships multiple weights/styles."""
    by_family: dict[str, tuple[str, bool]] = {}  # family → (path, is_regular)
    for d in _FONT_DIRS:
        if not d.exists():
            continue
        for path in d.iterdir():
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            if path.name.startswith("."):
                continue
            try:
                f = ImageFont.truetype(str(path), 12, index=0)
                family, style = f.getname()
            except Exception:
                continue
            if not family:
                continue
            is_regular = (style or "").lower() in ("", "regular", "book")
            entry = by_family.get(family)
            if entry is None or (is_regular and not entry[1]):
                by_family[family] = (str(path), is_regular)
    return {f: p for f, (p, _) in by_family.items()}


@lru_cache(maxsize=1)
def _installed_families() -> set[str]:
    """Return the set of font family names installed on the system."""
    return set(_family_to_path().keys())


def font_path(family: str) -> str | None:
    """Return the file path for an installed font family, or None."""
    return _family_to_path().get(family)


def list_curated_fonts() -> list[dict[str, Any]]:
    """Return the curated math/science fonts annotated with installed status."""
    installed = _installed_families()
    return [
        {**f, "installed": f["family"] in installed}
        for f in CURATED_FONTS
    ]


# ── Presets ────────────────────────────────────────────────────────────


def _installed_pool() -> list[str]:
    """Curated families that are actually installed, in CURATED_FONTS order."""
    installed = _installed_families()
    return [f["family"] for f in CURATED_FONTS if f["family"] in installed]


def get_preset(name: str) -> dict[str, Any]:
    """Return a SynthesisConfig dict for 'beginner', 'intermediate', or 'advanced'.

    Picks from the curated, installed pool (CURATED_FONTS ∩ installed). Slicing
    keeps train/val disjoint and degrades gracefully when fewer fonts are
    available than the preset would prefer.
    """
    pool = _installed_pool()

    if name == "beginner":
        train = pool[:2]
        val = pool[2:4]
        return {
            "name": "beginner",
            "categories": ["digits"],
            "training_fonts": train,
            "validation_fonts": val,
            "augmentation": {
                "noise": {"enabled": False, "max_level": 0},
                "skew": {"enabled": False},
            },
        }

    if name == "intermediate":
        train = pool[:6]
        val = pool[6:9]
        return {
            "name": "intermediate",
            "categories": ["digits", "lower_roman", "upper_roman", "punctuation"],
            "training_fonts": train,
            "validation_fonts": val,
            "augmentation": {
                "noise": {"enabled": True, "max_level": 25},
                "skew": {"enabled": False},
            },
        }

    if name == "advanced":
        # All installed curated fonts. ~80/20 train/val with evenly-spaced
        # indices so adjacent fonts in CURATED_FONTS don't all bunch into
        # the same partition.
        if len(pool) <= 1:
            train, val = pool, []
        else:
            n_val = max(2, len(pool) // 5)
            if n_val >= len(pool):
                n_val = len(pool) - 1
            step = len(pool) / n_val
            # Pick val indices at midpoints between training fonts so the
            # highest-priority font (index 0) stays in training.
            val_indices = {int(step / 2 + i * step) for i in range(n_val)}
            train = [f for i, f in enumerate(pool) if i not in val_indices]
            val = [f for i, f in enumerate(pool) if i in val_indices]
        return {
            "name": "advanced",
            "categories": list(SYMBOL_CATEGORIES.keys()),
            "training_fonts": train,
            "validation_fonts": val,
            "augmentation": {
                "noise": {"enabled": True, "max_level": 50},
                "skew": {"enabled": True},
            },
        }

    raise ValueError(f"unknown preset: {name}")

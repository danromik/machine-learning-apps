"""
Filesystem locations for the language model project

Single source of truth: every other module imports its paths from here
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data"             # raw corpus + tokenized .bin files
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"   # trained BPE vocab/merges 
CKPT_DIR = PROJECT_DIR / "checkpoints"      # standalone checkpoints
RUNS_DIR = PROJECT_DIR / "runs"             # one directory per training run

for _d in (DATA_DIR, TOKENIZER_DIR, CKPT_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "valid")

def raw_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {list(SPLITS)}")
    return DATA_DIR / f"TinyStoriesV2-GPT4-{split}.txt"

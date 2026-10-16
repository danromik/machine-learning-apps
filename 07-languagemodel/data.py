import urllib.request
from pathlib import Path
from paths import DATA_DIR, raw_path

import numpy as np
from tokenizer import load_tokenizer, END_OF_TEXT, DEFAULT_VOCAB_SIZE

import torch

from rich.progress import (
    Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
)

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

RAW_URLS = {
    "train" : f"{BASE}/TinyStoriesV2-GPT4-train.txt",
    "valid" : f"{BASE}/TinyStoriesV2-GPT4-valid.txt"
}

def download_raw(split: str, force: bool = False) -> Path:
    """Download one TinyStories split into DATA_DIR. Returns the file path"""
    dest = raw_path(split)

    if dest.exists() and not force:
        print(f"{dest.name} already present ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    part = dest.with_suffix(".part")
    with urllib.request.urlopen(RAW_URLS[split]) as response:
        length = response.headers.get("Content-Length")
        total = int(length) if length is not None else None

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"TinyStories {split}", total=total)

            with open(part, "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    part.replace(dest)
    return dest

def bin_path(split: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    return DATA_DIR / f"{split}-bpe{vocab_size}.bin"

def _flush_docs(tok, docs: list[str], eot_id: int, out) -> int:
    """Encode a batch of documents, appends ids to the open binary file"""
    ids: list[int] = []
    for encoding in tok.encode_batch(docs):
        ids.extend(encoding.ids)
        ids.append(eot_id)

    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out)
    return len(arr)

def encode_split(
    split: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    docs_per_batch: int = 1000,
    force: bool = False,
) -> Path:
    """Tokenize one raw split into a flat uint16 .bin of token ids"""
    dest = bin_path(split, vocab_size)

    if dest.exists() and not force:
        print(f"{dest.name} already present ({dest.stat().st_size // 2 / 1e6:.1f}M tokens)")
        return dest

    tok = load_tokenizer(vocab_size)
    eot_id = tok.token_to_id(END_OF_TEXT)

    src = raw_path(split)
    part = dest.with_suffix(".part")

    total_tokens = 0
    docs: list[str] = []
    lines: list[str] = []

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"encoding {split}", total=src.stat().st_size)

        with open(part, "wb") as out, open(src, "r", encoding="utf-8") as f:
            for line in f:
                progress.update(task, advance=len(line.encode("utf-8")))

                if line.strip() == END_OF_TEXT:
                    doc = "".join(lines).strip()
                    lines = []
                    if doc:
                        docs.append(doc)
                    if len(docs) >= docs_per_batch:
                        total_tokens += _flush_docs(tok, docs, eot_id, out)
                        docs = []
                else:
                    lines.append(line)

            tail = "".join(lines).strip()
            if tail:
                docs.append(tail)
            if docs:
                total_tokens += _flush_docs(tok, docs, eot_id, out)

    part.replace(dest)
    print(f"{dest.name}: {total_tokens / 1e6:.1f}M tokens")
    return dest

def load_tokens(split: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> np.memmap:
    """Open a tokenized split as a read-only memory-mapped uint16 array"""
    path = bin_path(split, vocab_size)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python data.py {split}")
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(
    tokens: np.memmap, batch_size: int, block_size: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) sequences"""
    ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(tokens[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])

    return x.to(device), y.to(device)

if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "valid"
    download_raw(split)
    encode_split(split)


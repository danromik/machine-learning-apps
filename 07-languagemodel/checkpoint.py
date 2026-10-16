import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    model_cfg,
    train_cfg,
    step: int,
    elapsed: float,
    best_val: float,
    history: list[dict],
) -> None:
    """Write a resumable checkpoint atomically"""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "step": step,
        "elapsed": elapsed,
        "best_val": best_val,
        "history": history,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }

    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint payload onto the CPU"""
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])

def rebuild(ckpt: dict, device: str):
    """Reconstruct model, optimizer, and both configs from a checkpoint"""
    from engine import TrainConfig
    from model import GPT, GPTConfig

    model_cfg = GPTConfig(**ckpt["model_config"])
    train_cfg = TrainConfig(**ckpt["train_config"])

    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    optimizer = model.configure_optimizers(train_cfg.weight_decay, train_cfg.lr)
    optimizer.load_state_dict(ckpt["optimizer"])

    restore_rng(ckpt["rng"])
    return model, optimizer, model_cfg, train_cfg


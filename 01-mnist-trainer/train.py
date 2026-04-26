"""CLI training script for MNIST.

Usage:
    uv run python 01-mnist-trainer/train.py --model cnn --epochs 5
    uv run python 01-mnist-trainer/train.py --model mlp --epochs 3
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from models import CKPT_DIR, build_model, get_loaders, pick_device


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_seen += x.size(0)
    return total_loss / total_seen


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CKPT_DIR.mkdir(exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch:2d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = CKPT_DIR / f"{args.model}_best.pt"
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                },
                ckpt,
            )
            print(f"  saved {ckpt.name}")

    print(f"best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()

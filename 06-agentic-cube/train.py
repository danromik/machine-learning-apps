"""Headless CLI trainer — the ground truth for the Cube RL pipeline.

Fastest way to confirm value iteration actually learns to solve the cube before
any server/UI exists. Trains on the reverse-scramble curriculum, ramping the
scramble depth `k` as the shallow depths are mastered, and periodically reports
solve-rate (the real metric) via beam search.

    uv run python 06-agentic-cube/train.py --cube 2 --iterations 1500
    uv run python 06-agentic-cube/train.py --cube 3 --iterations 8000 --device mps
    uv run python 06-agentic-cube/train.py --cube 2 --save pretrained-2x2.pt

Curriculum: starts at `--start-k` and promotes to the next depth whenever the
solve-rate at the current depth clears `--promote-at`, up to `--max-k`.
"""

from __future__ import annotations

import argparse

import torch

from agents import ALGORITHMS, default_hyperparams
from training import CubeSession, save_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="Train a cube solver (headless).")
    p.add_argument("--cube", type=int, choices=(2, 3), default=2, help="cube size")
    p.add_argument("--algo", choices=list(ALGORITHMS), default="value_iteration")
    p.add_argument("--iterations", type=int, default=1500, help="value-iteration batches")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--start-k", type=int, default=1)
    p.add_argument("--max-k", type=int, default=None, help="default 7 (2x2) / 14 (3x3)")
    p.add_argument("--promote-at", type=float, default=0.9,
                   help="solve-rate at current k that triggers promotion to k+1")
    p.add_argument("--eval-every", type=int, default=100, help="iterations between evals")
    p.add_argument("--eval-n", type=int, default=80)
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="checkpoint filename to save at the end")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        from models import pick_device
        device = pick_device()
    else:
        device = torch.device(args.device)

    max_k = args.max_k if args.max_k is not None else (7 if args.cube == 2 else 14)
    hp = default_hyperparams(args.algo)
    hp.update({"batch_size": args.batch_size, "lr": args.lr})
    session = CubeSession(args.algo, hp, {"size": args.cube}, device, seed=args.seed)
    session.current_k = args.start_k

    print(f"Training {ALGORITHMS[args.algo]['label']} on a {args.cube}x{args.cube} cube "
          f"for {args.iterations} iterations (device={device})")
    print(f"curriculum: k={args.start_k}..{max_k}, promote at solve-rate "
          f">= {args.promote_at:.0%}, state dim={session.in_features}\n")

    for it in range(1, args.iterations + 1):
        rec = session.train_one_iteration()
        if it % args.eval_every == 0 or it == 1:
            ev = session.evaluate(n=args.eval_n, k=session.current_k)
            rate = ev["solve_rate"]
            sol = ev["mean_solution_len"]
            print(f"it {it:5d}  k={session.current_k:2d}  loss={rec['loss']:.4f}  "
                  f"solve_rate={rate:5.1%}  mean_len={sol}")
            if rate >= args.promote_at and session.current_k < max_k:
                session.current_k += 1
                print(f"            ↑ promoted curriculum to k={session.current_k}")

    # Final eval sweep across depths.
    print("\nFinal solve-rate by scramble depth:")
    for k in range(1, max_k + 1):
        ev = session.evaluate(n=args.eval_n, k=k)
        print(f"  k={k:2d}: {ev['solve_rate']:5.1%}  (mean_len={ev['mean_solution_len']})")

    if args.save:
        name = save_checkpoint(session, args.save)
        print(f"\nsaved checkpoint -> checkpoints/{name}")


if __name__ == "__main__":
    main()

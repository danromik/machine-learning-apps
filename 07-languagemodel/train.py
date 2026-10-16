import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from checkpoint import load_checkpoint, rebuild, save_checkpoint
from console import Dashboard, KeyReader, format_time
from data import load_tokens
from engine import TrainConfig, estimate_loss, lr_at, set_lr, train_step
from model import GPT, GPTConfig, pick_device
from paths import RUNS_DIR
from tokenizer import DEFAULT_VOCAB_SIZE, END_OF_TEXT, load_tokenizer

from rich.live import Live


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small GPT on TinyStories.")

    p.add_argument("--name", default=None, help="run name (default: timestamp only)")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--device", default=None, help="override device selection")

    #model
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)

    #training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--sample-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=500)

    return p.parse_args()


def make_run_dir(name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / (f"{stamp}-{name}" if name else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def setup(args: argparse.Namespace, device: str):
    """Build a fresh model, or restore one from a checkpoint"""
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume))
        model, optimizer, model_cfg, train_cfg = rebuild(ckpt, device)

        # the schedule may legitimately be extended on resume
        train_cfg.max_steps = args.max_steps

        state = {
            "step": ckpt["step"],
            "elapsed": ckpt["elapsed"],
            "best_val": ckpt["best_val"],
            "history": ckpt["history"],
        }
        return model, optimizer, model_cfg, train_cfg, state

    model_cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        checkpoint_every=args.checkpoint_every,
    )

    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(train_cfg.weight_decay, train_cfg.lr)

    state = {"step": 0, "elapsed": 0.0, "best_val": float("inf"), "history": []}

    return model, optimizer, model_cfg, train_cfg, state

def sample_text(
        model,
        tok,
        device: str,
        prompt: str = "Once upon a time",
        max_new_tokens: int = 120,
) -> str:
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=200,
        eot_id=tok.token_to_id(END_OF_TEXT),
    )
    return tok.decode(out[0].tolist())

def write_metrics(run_dir: Path, record: dict) -> None:
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def save(path: Path, model, optimizer, model_cfg, train_cfg, state: dict) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        step=state["step"],
        elapsed=state["elapsed"],
        best_val=state["best_val"],
        history=state["history"],
    )

KEY_HELP = "space pause · s save · g sample · e eval · +/- lr · q quit"


def handle_keys(keys: KeyReader, ctl: dict, train_cfg: TrainConfig) -> None:
    """Drain pending keypresses and fold them into the control dict"""
    while (key := keys.get()) is not None:
        if key == " ":
            ctl["paused"] = not ctl["paused"]
        elif key == "q":
            ctl["quit"] = True
        elif key == "s":
            ctl["save"] = True
        elif key == "g":
            ctl["sample"] = True
        elif key == "e":
            ctl["eval"] = True
        elif key in "+=":
            train_cfg.lr *= 1.5
        elif key in "-_":
            train_cfg.lr /= 1.5


def main() -> None:
    args = parse_args()
    device = args.device or pick_device()
    run_dir = make_run_dir(args.name)

    tok = load_tokenizer(args.vocab_size)
    train_tokens = load_tokens("train", args.vocab_size)
    val_tokens = load_tokens("valid", args.vocab_size)

    model, optimizer, model_cfg, train_cfg, state = setup(args, device)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "device": device,
                "resumed_from": args.resume,
            },
            f,
            indent=2,
        )


    dash = Dashboard(train_cfg.max_steps, title=run_dir.name)
    ctl = {"paused": False, "quit": False, "save": False, "sample": False, "eval": False}

    val_loss = float("nan")
    elapsed_base = state["elapsed"]
    session_steps = 0
    t0 = time.time()

    with KeyReader() as keys, Live(dash.render(), refresh_per_second=8) as live:
        while state["step"] < train_cfg.max_steps and not ctl["quit"]:
            handle_keys(keys, ctl, train_cfg)

            if ctl["paused"]:
                dash.status = f"PAUSED · {KEY_HELP}"
                live.update(dash.render())
                time.sleep(0.05)
                continue

            lr = lr_at(
                state["step"],
                lr=train_cfg.lr,
                warmup_steps=train_cfg.warmup_steps,
                max_steps=train_cfg.max_steps,
                min_lr_frac=train_cfg.min_lr_frac,
            )
            set_lr(optimizer, lr)

            loss, norm = train_step(
                model, optimizer, train_tokens, train_cfg.batch_size, model_cfg.block_size,
                device, train_cfg.grad_clip
            )

            state["step"] += 1
            session_steps += 1
            state["elapsed"] = elapsed_base + (time.time() - t0)
            step = state["step"]

            dash.advance(step, loss)
            tps = session_steps * train_cfg.batch_size * model_cfg.block_size / (time.time() - t0)
            dash.stats = {
                "step": f"{step}/{train_cfg.max_steps}",
                "train loss": f"{loss:.3f}",
                "val loss": f"{val_loss:.3f}",
                "best val": f"{state['best_val']:.3f}",
                "lr": f"{lr:.2e}",
                "grad norm": f"{norm:.2f}",
                "tok/s": f"{tps:,.0f}",
                "elapsed": format_time(state["elapsed"]),
            }

            if ctl["eval"] or step % train_cfg.eval_every == 0:
                ctl["eval"] = False
                dash.status = "evaluating..."
                live.update(dash.render())

                val_loss = estimate_loss(
                    model, val_tokens, train_cfg.batch_size,
                    model_cfg.block_size, device, train_cfg.eval_batches,
                )
                state["history"].append({"step": step, "val_loss": val_loss})
                write_metrics(run_dir, {
                    "step": step, "train_loss": loss, "val_loss": val_loss,
                    "lr": lr, "elapsed": state["elapsed"],
                })

                if val_loss < state["best_val"]:
                    state["best_val"] = val_loss
                    save(run_dir / "ckpt-best.pt", model, optimizer, model_cfg, train_cfg, state)

            if ctl["sample"] or step % train_cfg.sample_every == 0:
                ctl["sample"] = False
                dash.status = "sampling..."
                live.update(dash.render())

                dash.sample = sample_text(model, tok, device)
                with open(run_dir / "samples.txt", "a") as f:
                    f.write(f"\n=== step {step} ===\n{dash.sample}\n")

            if ctl["save"] or step % train_cfg.checkpoint_every == 0:
                ctl["save"] = False
                save(run_dir / "ckpt-last.pt", model, optimizer, model_cfg, train_cfg, state)

            dash.status = KEY_HELP
            live.update(dash.render())

    save(run_dir / "ckpt-last.pt", model, optimizer, model_cfg, train_cfg, state)
    print(f"\nsaved to {run_dir}")


if __name__ == "__main__":
    main()
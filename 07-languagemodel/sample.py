import argparse
from pathlib import Path

import torch

from checkpoint import load_checkpoint, rebuild
from model import pick_device
from tokenizer import END_OF_TEXT, load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    p.add_argument("checkpoint", help="path to a .pt checkpoint")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    device = args.device or pick_device()

    ckpt = load_checkpoint(Path(args.checkpoint))
    model, _, model_cfg, _ = rebuild(ckpt, device)
    model.eval()

    # rebuild() restores the training RNG state, so seed afterwards
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tok = load_tokenizer(model_cfg.vocab_size)
    eot_id = tok.token_to_id(END_OF_TEXT)
    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)

    print(f"checkpoint step {ckpt['step']}  ·  val loss {ckpt['best_val']:.3f}\n")

    for i in range(args.num_samples):
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eot_id=eot_id,
        )
        if args.num_samples > 1:
            print(f"--- sample {i+1} ---")
        print(tok.decode(out[0].tolist()) + "\n")


if __name__ == "__main__":
    main()

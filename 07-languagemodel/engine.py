import math
import torch
from dataclasses import dataclass
from data import get_batch

@dataclass
class TrainConfig:
    # optimization
    batch_size: int = 32
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # cadence
    eval_every: int = 250
    eval_batches: int = 50
    sample_every: int = 250
    checkpoint_every: int = 500

    # misc
    seed: int = 1337
    compile: bool = False


def lr_at(
    step: int,
    *,
    lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_frac: float = 0.1,
) -> float:
    """Linear warmup, then cosine decay to min_lr_frac * lr"""
    if step < warmup_steps:
        return lr * (step + 1) / (warmup_steps + 1)
    if step >= max_steps:
        return lr * min_lr_frac

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_frac + (1.0 - min_lr_frac) * coeff)

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr

def train_step(
    model,
    optimizer: torch.optim.Optimizer,
    tokens,
    batch_size: int,
    block_size: int,
    device: str,
    grad_clip: float,
) -> tuple[float, float]:
    """One gradient step. Returns (loss, grad_norm),"""
    x, y = get_batch(tokens, batch_size, block_size, device)

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), norm.item()

@torch.no_grad()
def estimate_loss(
    model,
    tokens,
    batch_size: int,
    block_size: int,
    device: str,
    batches: int,
) -> float:
    """Average loss over several batches, with the model in eval mode"""
    was_training = model.training
    model.eval()

    total = 0.0
    for _ in range(batches):
        x, y = get_batch(tokens, batch_size, block_size, device)
        _, loss = model(x, y)
        total += loss.item()

    model.train(was_training)
    return total / batches


if __name__ == "__main__":
    import time
    from data import load_tokens
    from model import GPT, GPTConfig, pick_device
    from tokenizer import END_OF_TEXT, load_tokenizer

    model_cfg = GPTConfig()
    cfg = TrainConfig(max_steps=200, warmup_steps=20)

    torch.manual_seed(cfg.seed)
    device = pick_device()

    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.lr)
    tokens = load_tokens("train")

    print(f"device {device}   params {model.num_params() / 1e6:.2f}M")

    t0 = time.time()
    for step in range(cfg.max_steps):
        lr = lr_at(
            step,
            lr=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
            min_lr_frac=cfg.min_lr_frac,
        )
        set_lr(optimizer, lr)

        loss, norm = train_step(
            model, optimizer, tokens,
            cfg.batch_size, model_cfg.block_size, device, cfg.grad_clip,
        )

        if step % 20 == 0 or step == cfg.max_steps - 1:
            dt = time.time() - t0
            tps = (step + 1) * cfg.batch_size * model_cfg.block_size / dt
            print(f"step {step:4d}  loss {loss:.3f}  |g| {norm:5.2f}  {tps:>8,.0f} tok/s")

    tok = load_tokenizer()
    prompt = "Once upon a time"
    ids = torch.tensor([tok.encode(prompt).ids], device=device)

    out = model.generate(
        ids,
        max_new_tokens=150,
        temperature=0.8,
        top_k=200,
        eot_id=tok.token_to_id(END_OF_TEXT)
    )
    print("\n" + tok.decode(out[0].tolist()))

    

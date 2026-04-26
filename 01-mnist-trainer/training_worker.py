"""Runs a PyTorch training loop in a background thread and emits
step/epoch events onto a queue for the UI to consume.

The worker keeps a *session* alive between runs — model, optimizer,
data-loader iterator, and step counter. That way "single batch", "single
epoch", and "continuous" clicks all advance the same training state. The
session is rebuilt from scratch when the model or batch size changes.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models import CKPT_DIR, build_model, get_loaders, pick_device


@dataclass
class TrainConfig:
    model: str
    epochs: int
    batch_size: int
    lr: float
    seed: int


class TrainingWorker:
    def __init__(self) -> None:
        self.events: queue.Queue[dict] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session: dict | None = None
        self._best_acc: float = 0.0
        self.auto_save_checkpoint: bool = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Public API ──────────────────────────────────────────────────────

    def start_run(
        self,
        cfg: TrainConfig,
        max_steps: int | None = None,
        max_epochs: int | None = None,
    ) -> None:
        """Start a background run.
        max_steps=None and max_epochs=None means run until cfg.epochs exhausts.
        Whichever cap is hit first ends the run.
        """
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(cfg, max_steps, max_epochs), daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def reset_session(self, cfg: TrainConfig | None = None) -> None:
        """Drop the current session. If cfg is given, immediately build a fresh
        random-weights session so prediction uses the new model right away
        (instead of the lazy build-on-next-run behavior). Safe to call while idle.
        """
        self._session = None
        self._best_acc = 0.0
        if cfg is not None:
            self._init_session(cfg, emit_events=False)

    def save_checkpoint(self, filename: str | None = None) -> str | None:
        """Manually save the current session's model. Returns the filename, or None if no session."""
        if self._session is None:
            return None
        cfg: TrainConfig = self._session["cfg"]
        s = self._session
        CKPT_DIR.mkdir(exist_ok=True)
        name = filename or f"{cfg.model}_manual.pt"
        path = CKPT_DIR / name
        torch.save(
            {
                "model": cfg.model,
                "state_dict": s["model"].state_dict(),
                "step": s["step"],
                "epoch": s["epoch"],
                "best_acc": self._best_acc,
            },
            path,
        )
        return path.name

    def load_into_session(self, ckpt: dict, cfg: TrainConfig) -> dict:
        """Build a training session from a loaded checkpoint dict.
        `cfg.model` is ignored — the checkpoint's stored model name wins.
        Returns {model, step, epoch, best_acc} for the caller to relay back.
        """
        model_name = ckpt.get("model", cfg.model)
        cfg = TrainConfig(
            model=model_name,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            seed=cfg.seed,
        )
        torch.manual_seed(cfg.seed)
        device = pick_device()
        train_loader, test_loader = get_loaders(cfg.batch_size)
        model = build_model(cfg.model).to(device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self._session = {
            "cfg": cfg,
            "device": device,
            "model": model,
            "optimizer": optimizer,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_iter": iter(train_loader),
            "steps_per_epoch": len(train_loader),
            "step": int(ckpt.get("step", 0)),
            "epoch": int(ckpt.get("epoch", 1)),
            "epoch_start_time": time.time(),
        }
        self._best_acc = float(ckpt.get("best_acc", 0.0))
        self._emit(type="reset")
        self._emit(
            type="log",
            msg=f"loaded · {cfg.model} · step {self._session['step']} · best_acc {self._best_acc:.4f}",
        )
        return {
            "model": cfg.model,
            "step": self._session["step"],
            "epoch": self._session["epoch"],
            "best_acc": self._best_acc,
        }

    def session_state(self) -> dict:
        """Snapshot of current session state for the UI on page load."""
        if self._session is None:
            return {
                "has_session": False,
                "model": None,
                "step": 0,
                "epoch": 1,
                "best_acc": 0.0,
                "running": self.running,
            }
        return {
            "has_session": True,
            "model": self._session["cfg"].model,
            "step": self._session["step"],
            "epoch": self._session["epoch"],
            "best_acc": self._best_acc,
            "running": self.running,
        }

    # ── Internals ───────────────────────────────────────────────────────

    def _emit(self, **kw) -> None:
        self.events.put(kw)

    def _need_new_session(self, cfg: TrainConfig) -> bool:
        if self._session is None:
            return True
        old: TrainConfig = self._session["cfg"]
        return old.model != cfg.model or old.batch_size != cfg.batch_size

    def _init_session(self, cfg: TrainConfig, emit_events: bool = True) -> None:
        torch.manual_seed(cfg.seed)
        device = pick_device()
        train_loader, test_loader = get_loaders(cfg.batch_size)
        model = build_model(cfg.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self._session = {
            "cfg": cfg,
            "device": device,
            "model": model,
            "optimizer": optimizer,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "train_iter": iter(train_loader),
            "steps_per_epoch": len(train_loader),
            "step": 0,
            "epoch": 1,
            "epoch_start_time": time.time(),
        }
        self._best_acc = 0.0
        if emit_events:
            self._emit(type="reset")
            self._emit(type="log", msg=f"new session · {cfg.model} · device {device}")

    def _sync_optimizer_lr(self, cfg: TrainConfig) -> None:
        """LR can change between runs without rebuilding the session."""
        for pg in self._session["optimizer"].param_groups:
            pg["lr"] = cfg.lr

    def _run(
        self,
        cfg: TrainConfig,
        max_steps: int | None,
        max_epochs: int | None,
    ) -> None:
        try:
            if self._need_new_session(cfg):
                self._init_session(cfg)
            else:
                self._sync_optimizer_lr(cfg)

            s = self._session
            model = s["model"]
            opt = s["optimizer"]
            device = s["device"]

            self._emit(
                type="start",
                steps_per_epoch=s["steps_per_epoch"],
                total_steps=s["steps_per_epoch"] * cfg.epochs,
                epochs=cfg.epochs,
                starting_step=s["step"],
                starting_epoch=s["epoch"],
                max_steps=max_steps,
                max_epochs=max_epochs,
            )

            steps_done_this_run = 0
            epochs_done_this_run = 0
            emit_every = 1 if max_steps == 1 else 10

            while True:
                if self._stop.is_set():
                    self._emit(type="stopped")
                    return
                if max_steps is not None and steps_done_this_run >= max_steps:
                    self._emit(type="paused", step=s["step"], epoch=s["epoch"])
                    return
                if max_epochs is not None and epochs_done_this_run >= max_epochs:
                    self._emit(type="paused", step=s["step"], epoch=s["epoch"])
                    return
                if s["epoch"] > cfg.epochs:
                    self._emit(type="done", best_acc=self._best_acc)
                    return

                try:
                    x, y = next(s["train_iter"])
                except StopIteration:
                    # End of epoch → evaluate, maybe checkpoint, advance.
                    val_loss, val_acc = _evaluate(model, s["test_loader"], device)
                    dt = time.time() - s["epoch_start_time"]
                    self._emit(
                        type="epoch",
                        step=s["step"],
                        epoch=s["epoch"],
                        val_loss=val_loss,
                        val_acc=val_acc,
                        seconds=dt,
                    )
                    if val_acc > self._best_acc:
                        self._best_acc = val_acc
                        if self.auto_save_checkpoint:
                            ckpt_path = CKPT_DIR / f"{cfg.model}_best.pt"
                            CKPT_DIR.mkdir(exist_ok=True)
                            torch.save(
                                {
                                    "model": cfg.model,
                                    "state_dict": model.state_dict(),
                                    "step": s["step"],
                                    "epoch": s["epoch"],
                                    "best_acc": self._best_acc,
                                },
                                ckpt_path,
                            )
                            self._emit(type="checkpoint", name=ckpt_path.name, val_acc=val_acc)
                    s["epoch"] += 1
                    s["epoch_start_time"] = time.time()
                    s["train_iter"] = iter(s["train_loader"])
                    epochs_done_this_run += 1
                    continue

                model.train()
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                s["step"] += 1
                steps_done_this_run += 1

                if s["step"] % emit_every == 0 or steps_done_this_run == max_steps:
                    self._emit(
                        type="step",
                        step=s["step"],
                        epoch=s["epoch"],
                        train_loss=float(loss.item()),
                    )
        except Exception as e:  # noqa: BLE001
            self._emit(type="error", msg=f"{type(e).__name__}: {e}")


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, float]:
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

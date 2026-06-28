"""The background training run — the piece that lets an overnight run exist.

Unlike the short, caller-driven training chunks (UI loop / agent tool), a
*background run* is a process-owned daemon thread that keeps training
independent of any chat turn or browser connection, checkpoints itself
periodically (so it's resumable if the server restarts), and can be stopped at
any time. It advances the **reverse-scramble curriculum** automatically: train
at depth `k`, evaluate solve-rate, and promote to `k+1` once the bar is cleared.

Concurrency model (see CLAUDE.md / the plan):
  * The training loop is a `threading.Thread` (the value-iteration loop is
    GIL-bound; an asyncio task would stall the WS sender between yields).
  * Every `CubeSession` mutation/read is guarded by a shared `threading.RLock`
    (`SESSION_LOCK`) so an autonomous check-in tool can safely read/tweak the
    session between chunks. The trainer releases the lock every iteration.
  * The thread reaches the WebSocket **only** via
    `agent_state.broadcast_event_threadsafe` (the asyncio queues are loop-affine).

Milestones (k-promotion, new checkpoint, solve-rate plateau, finish) are recorded
on the status snapshot so the check-in scheduler can wake the RL Coach on them.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

import agent_state
import training as _training
from models import CKPT_DIR

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

# Shared lock guarding the live CubeSession across the trainer thread and the
# (event-loop-thread) REST handlers / MCP tools.
SESSION_LOCK = threading.RLock()

# How often (seconds) to push a throttled trainer_progress event.
_PROGRESS_INTERVAL = 1.0


@dataclass
class RunConfig:
    cube_size: int = 3
    start_k: int = 1
    max_k: int = 14
    promote_at: float = 0.9
    max_iterations: int = 0      # 0 = unbounded (until stopped)
    max_wall_seconds: float = 0  # 0 = unbounded
    eval_every: int = 100
    eval_n: int = 80
    checkpoint_every: int = 200  # iterations between auto-checkpoints
    cadence_minutes: float = 20  # RL Coach check-in time floor
    coach_session_id: str | None = None


def _autorun_name(run_id: str) -> str:
    return f"autorun_{run_id}.pt"


def _manifest_path(run_id: str) -> Path:
    return REPORTS_DIR / f"run_{run_id}.manifest.json"


class BackgroundTrainer:
    def __init__(
        self,
        get_session: Callable[[], object | None],
        set_session: Callable[[object | None], None],
        get_device: Callable[[], torch.device],
        build_session_from_state: Callable[[], object],
    ):
        self._get_session = get_session
        self._set_session = set_session
        self._get_device = get_device
        self._build_session = build_session_from_state

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cfg: RunConfig | None = None
        self.run_id: str | None = None
        self.state: str = "idle"  # idle|running|stopping|stopped|finished|error
        self.error: str | None = None
        self.started_at: float | None = None
        self.last_checkpoint: str | None = None
        self.last_checkpoint_at: float | None = None
        # Milestone signalling for the check-in scheduler.
        self.milestone: str | None = None
        self.milestone_at: float | None = None
        self._last_progress_emit = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: RunConfig, *, fresh: bool = True, run_id: str | None = None) -> dict:
        if self.is_running():
            raise RuntimeError("a training run is already in progress")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._cfg = cfg
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self._stop.clear()
        self.error = None
        self.milestone = None
        self.started_at = time.time()

        # Build or reuse the session under the lock.
        with SESSION_LOCK:
            session = self._get_session()
            if fresh or session is None or getattr(session, "size", None) != cfg.cube_size:
                session = self._build_session()
                self._set_session(session)
            session.current_k = max(1, int(cfg.start_k))

        self.state = "running"
        self._thread = threading.Thread(target=self._run, name="cube-trainer", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, timeout: float = 10.0) -> dict:
        if self.is_running():
            self.state = "stopping"
            self._stop.set()
            self._thread.join(timeout=timeout)
        return self.status()

    def update_curriculum(self, *, max_k: int | None = None, promote_at: float | None = None) -> None:
        """Live-adjust the running run's curriculum (read by the loop next eval)."""
        if self._cfg is None:
            return
        if max_k is not None:
            self._cfg.max_k = max(1, int(max_k))
        if promote_at is not None:
            self._cfg.promote_at = float(promote_at)

    def _signal_milestone(self, reason: str) -> None:
        self.milestone = reason
        self.milestone_at = time.time()

    def consume_milestone(self) -> str | None:
        """Read-and-clear the pending milestone (called by the scheduler)."""
        m = self.milestone
        self.milestone = None
        return m

    # ── The training loop (runs on the daemon thread) ────────────────────────
    def _run(self) -> None:
        cfg = self._cfg
        assert cfg is not None
        try:
            agent_state.broadcast_event_threadsafe(
                {"type": "trainer_status", "source": "trainer", "state": "running",
                 "run_id": self.run_id}
            )
            self._signal_milestone("run_started")
            while not self._stop.is_set():
                if cfg.max_iterations and self._iteration() >= cfg.max_iterations:
                    break
                if cfg.max_wall_seconds and (time.time() - self.started_at) >= cfg.max_wall_seconds:
                    break

                with SESSION_LOCK:
                    session = self._get_session()
                    if session is None:
                        break
                    rec = session.train_one_iteration()
                    it = session.current_k  # noqa: F841 (kept for clarity)

                self._maybe_emit_progress(rec)

                it_count = self._iteration()
                if cfg.eval_every and it_count % cfg.eval_every == 0:
                    self._do_eval(cfg)
                if cfg.checkpoint_every and it_count % cfg.checkpoint_every == 0:
                    self._do_checkpoint()

            # Clean finish or stop.
            self._do_checkpoint()
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # noqa: BLE001 — surface any trainer crash
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = "error"
        finally:
            agent_state.broadcast_event_threadsafe(
                {"type": "trainer_status", "source": "trainer", "state": self.state,
                 "run_id": self.run_id, "error": self.error}
            )
            self._signal_milestone(f"run_{self.state}")

    def _iteration(self) -> int:
        s = self._get_session()
        return getattr(s, "iteration", 0) if s is not None else 0

    def _maybe_emit_progress(self, rec: dict) -> None:
        now = time.time()
        if now - self._last_progress_emit < _PROGRESS_INTERVAL:
            return
        self._last_progress_emit = now
        s = self._get_session()
        agent_state.broadcast_event_threadsafe({
            "type": "trainer_progress",
            "source": "trainer",
            "record": rec,
            "iteration": getattr(s, "iteration", 0),
            "current_k": getattr(s, "current_k", 1),
            "solve_rate_by_k": dict(getattr(s, "solve_rate_by_k", {})),
        })

    def _do_eval(self, cfg: RunConfig) -> None:
        with SESSION_LOCK:
            session = self._get_session()
            if session is None:
                return
            k = session.current_k
            ev = session.evaluate(n=cfg.eval_n, k=k)
            promoted = False
            if ev["solve_rate"] >= cfg.promote_at and session.current_k < cfg.max_k:
                session.current_k += 1
                promoted = True
            new_k = session.current_k
        agent_state.broadcast_event_threadsafe({
            "type": "trainer_status", "source": "trainer", "state": "eval",
            "run_id": self.run_id, "eval": ev, "current_k": new_k, "promoted": promoted,
        })
        if promoted:
            self._signal_milestone(f"promoted_k_{new_k}")
        if new_k >= cfg.max_k and ev["solve_rate"] >= cfg.promote_at:
            self._signal_milestone("max_k_reached")

    def _do_checkpoint(self) -> None:
        with SESSION_LOCK:
            session = self._get_session()
            if session is None:
                return
            name = _training.save_checkpoint(session, _autorun_name(self.run_id))
            it = session.iteration
            k = session.current_k
        self.last_checkpoint = name
        self.last_checkpoint_at = time.time()
        self._write_manifest(it, k)
        agent_state.broadcast_event_threadsafe({
            "type": "trainer_status", "source": "trainer", "state": "checkpoint",
            "run_id": self.run_id, "checkpoint": name, "iteration": it,
        })
        self._signal_milestone("checkpoint")

    def _write_manifest(self, iteration: int, current_k: int) -> None:
        cfg = self._cfg
        manifest = {
            "run_id": self.run_id,
            "config": asdict(cfg) if cfg else {},
            "iteration": iteration,
            "current_k": current_k,
            "checkpoint": self.last_checkpoint,
            "state": self.state,
            "updated_at": time.time(),
        }
        try:
            _manifest_path(self.run_id).write_text(json.dumps(manifest, indent=2))
        except OSError:
            pass

    # ── Status snapshot ──────────────────────────────────────────────────────
    def status(self) -> dict:
        s = self._get_session()
        cfg = self._cfg
        return {
            "state": self.state,
            "running": self.is_running(),
            "run_id": self.run_id,
            "error": self.error,
            "started_at": self.started_at,
            "iteration": getattr(s, "iteration", 0) if s is not None else 0,
            "current_k": getattr(s, "current_k", 1) if s is not None else 1,
            "solve_rate_by_k": dict(getattr(s, "solve_rate_by_k", {})) if s is not None else {},
            "last_checkpoint": self.last_checkpoint,
            "last_checkpoint_at": self.last_checkpoint_at,
            "config": asdict(cfg) if cfg else None,
        }


# ── Resume detection ─────────────────────────────────────────────────────────
def find_resumable() -> dict | None:
    """Return the most recent unfinished run manifest (state not finished), if
    its checkpoint still exists — so the server can offer to resume it."""
    if not REPORTS_DIR.exists():
        return None
    best: dict | None = None
    for p in sorted(REPORTS_DIR.glob("run_*.manifest.json")):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("state") in ("finished",):
            continue
        ckpt = m.get("checkpoint")
        if not ckpt or not (CKPT_DIR / ckpt).exists():
            continue
        if best is None or m.get("updated_at", 0) > best.get("updated_at", 0):
            best = m
    return best

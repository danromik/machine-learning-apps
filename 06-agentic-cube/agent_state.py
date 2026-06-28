"""Backend mirror of the pipeline state that the RL Coach agent and the
frontend Svelte stores both read and write through.

Same design as the other agentic apps: one `pipeline_state` dict, an async lock
around mutations, and a fan-out broadcaster pushing deep-merge patches to every
subscribed WebSocket. Here the state describes a *cube* RL pipeline: the
environment (cube size + reverse-scramble curriculum), the algorithm + its
hyperparameters, and training preferences.

Transient progress (trainer ticks, session resets, report updates, bridged
autonomous agent-turn events) rides the same socket via `broadcast_event` /
`broadcast_event_threadsafe` but bypasses the deep-merge state model.

`broadcast_event_threadsafe` is the **only** path the background-training thread
may use to reach the socket: the subscriber queues are `asyncio.Queue`s, which
are loop-affine and not thread-safe, so the thread hops onto the event loop via
`loop.call_soon_threadsafe`.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any, AsyncIterator

from agents import default_hyperparams

DEFAULT_ALGO = "value_iteration"


# ── Default state ──────────────────────────────────────────────────────
# Mirrors the frontend defaults in state.svelte.ts so the UI matches the
# backend on first load, and wire-format matches store shapes (no
# translation in the WS layer).
def _default_state() -> dict[str, Any]:
    return {
        "environment": {
            "size": 3,  # 2 (Pocket Cube) or 3 (standard)
            "curriculum": {
                "startK": 1,
                "maxK": 14,
                "promoteAt": 0.9,   # solve-rate at current k that promotes to k+1
            },
        },
        "algorithm": {
            "algo": DEFAULT_ALGO,
            "hyperparameters": default_hyperparams(DEFAULT_ALGO),
        },
        "training": {
            "iterationsPerRun": 200,   # the "how many iterations" UI control
            "evalEveryN": 100,          # solve-rate eval cadence during a run
            "evalN": 80,                # scrambles per eval
            "cadenceMinutes": 20,       # RL Coach autonomous check-in time floor
        },
    }


# ── Module state ───────────────────────────────────────────────────────

pipeline_state: dict[str, Any] = _default_state()

# asyncio.Lock — created lazily so this module is importable from sync code
# (server.py at module level) without a running loop.
_state_lock: asyncio.Lock | None = None

# The running event loop, captured at startup, so the background trainer thread
# can broadcast onto it from outside the loop.
_loop: asyncio.AbstractEventLoop | None = None


def _lock() -> asyncio.Lock:
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture the event loop (called from a FastAPI startup hook)."""
    global _loop
    _loop = loop


# Subscribers: each WS client gets its own queue; broadcasting fans out to
# every queue. Slow clients block only themselves.
_subscribers: list[asyncio.Queue] = []


# ── Patch / broadcast ──────────────────────────────────────────────────

def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """In-place deep merge; dict values recurse, everything else overwrites."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _snapshot() -> dict[str, Any]:
    return copy.deepcopy(pipeline_state)


async def apply_patch(patch: dict[str, Any], *, source: str = "unknown") -> dict[str, Any]:
    """Deep-merge a patch into pipeline_state and broadcast it. `source`
    tags the originator ("agent" | "ui" | "system") so subscribers can
    ignore echoes of their own edits."""
    async with _lock():
        _deep_merge(pipeline_state, patch)
    msg = {"type": "state_patch", "source": source, "patch": patch}
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass
    return patch


async def replace_state(new_state: dict[str, Any], *, source: str = "system") -> None:
    """Wholesale replacement (used when the whole pipeline reconfigures at
    once, e.g. load_checkpoint). Broadcasts a state_replace snapshot."""
    async with _lock():
        pipeline_state.clear()
        pipeline_state.update(copy.deepcopy(new_state))
    msg = {"type": "state_replace", "source": source, "state": _snapshot()}
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def get_state() -> dict[str, Any]:
    return _snapshot()


def broadcast_event(ev: dict[str, Any]) -> None:
    """Push a custom event to all WS subscribers without mutating
    pipeline_state — used for live training progress (trainer ticks, session
    resets, report updates) that don't fit the deep-merge model. Must be called
    on the event loop."""
    for q in list(_subscribers):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass


def broadcast_event_threadsafe(ev: dict[str, Any]) -> None:
    """Thread-safe `broadcast_event` for the background trainer thread. Hops
    onto the captured event loop; a no-op if the loop isn't set yet."""
    loop = _loop
    if loop is None:
        return
    loop.call_soon_threadsafe(broadcast_event, ev)


# ── Subscription ───────────────────────────────────────────────────────

def subscribe(maxsize: int = 512) -> tuple[asyncio.Queue, Any]:
    """Register a subscriber. Returns (queue, unsubscribe_fn)."""
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _subscribers.append(q)

    def unsubscribe() -> None:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass

    return q, unsubscribe


async def stream_events(queue: asyncio.Queue) -> AsyncIterator[dict[str, Any]]:
    while True:
        ev = await queue.get()
        yield ev

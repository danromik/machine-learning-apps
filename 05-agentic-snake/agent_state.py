"""Backend mirror of the pipeline state that the RL Coach agent and the
frontend Svelte stores both read and write through.

Same design as `04-agentic-symbols`: one `pipeline_state` dict, an async lock
around mutations, and a fan-out broadcaster pushing deep-merge patches to
every subscribed WebSocket. The only difference is the *shape* of the state —
here it describes an RL pipeline (environment + algorithm + training prefs)
rather than a synthesis/architecture one.

Transient training progress (per-episode ticks, session resets) rides the
same socket via `broadcast_event` but bypasses the deep-merge state model.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any, AsyncIterator

from agents import default_hyperparams


# ── Default state ──────────────────────────────────────────────────────
# Mirrors the frontend defaults in state.svelte.ts so the UI matches the
# backend on first load, and wire-format matches store shapes (no
# translation in the WS layer).
def _default_state() -> dict[str, Any]:
    return {
        "environment": {
            "width": 10,
            "height": 10,
            "observation": "features",
            "reward": {
                "food": 1.0,
                "death": -1.0,
                "step": 0.0,
                "toward_food": 0.0,
                "away_from_food": 0.0,
            },
        },
        "algorithm": {
            "algo": "qlearning",
            "hyperparameters": default_hyperparams("qlearning"),
        },
        "training": {
            "episodesPerRun": 200,   # the "how many episodes" control
            "evalEveryN": 25,         # greedy-eval cadence during a run
        },
    }


# ── Module state ───────────────────────────────────────────────────────

pipeline_state: dict[str, Any] = _default_state()

# asyncio.Lock — created lazily so this module is importable from sync code
# (server.py at module level) without a running loop.
_state_lock: asyncio.Lock | None = None


def _lock() -> asyncio.Lock:
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


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
    pipeline_state — used for live training progress (episode ticks,
    session resets) that don't fit the deep-merge model."""
    for q in list(_subscribers):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass


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

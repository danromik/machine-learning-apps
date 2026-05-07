"""Backend mirror of the pipeline state that the ML Engineer agent and
the frontend Svelte stores both read and write through.

Why this exists: Math Symbol Trainer kept the pipeline config (synthesis,
architecture, training prefs) entirely in the browser — every backend call
got the relevant slice as request payload. That works for one driver. But
once the agent can also drive, both sides need to see the same state and
both sides need to be notified when the other one changes it.

Design: one `pipeline_state` dict, one async lock around mutations, a
fan-out broadcaster that pushes patches to every subscribed WebSocket
queue. Patches are deep-merge dicts — `{"synthesis": {"augmentation":
{"noise": {"max_level": 30}}}}` — so the frontend can just merge them
into its store without per-field switch logic.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any, AsyncIterator


# ── Default state ──────────────────────────────────────────────────────

# Initial values mirror the frontend defaults in state.svelte.ts so the
# UI looks identical the first time it loads against this backend, and
# wire-format matches frontend store shapes (no translation in the WS
# layer).
def _default_state() -> dict[str, Any]:
    return {
        "synthesis": {
            # category id → bool. True means "include in training".
            "selectedCategories": {},
            # font family → "off" | "train" | "val".
            "fontUsage": {},
            "augmentation": {
                "noise": {"enabled": False, "max_level": 25},
                "skew": {"enabled": False},
            },
            "activePreset": None,
        },
        "architecture": {
            "layers": [],
            "hyperparameters": {
                "lr": 0.001,
                "batch_size": 128,
                "optimizer": "adam",
            },
        },
        "training": {
            "validateEveryN": 10,
            "samplesPerSymbolPerEpoch": 50,
        },
    }


# ── Module state ───────────────────────────────────────────────────────

pipeline_state: dict[str, Any] = _default_state()

# asyncio.Lock — created lazily so this module is importable from sync
# code (server.py at module level) without a running loop.
_state_lock: asyncio.Lock | None = None


def _lock() -> asyncio.Lock:
    global _state_lock
    if _state_lock is None:
        _state_lock = asyncio.Lock()
    return _state_lock


# Subscribers: each WS client gets its own asyncio.Queue. Broadcasting
# fans out one patch to every queue. Slow clients block themselves only.
_subscribers: list[asyncio.Queue] = []


# ── Patch / broadcast ──────────────────────────────────────────────────


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """In-place deep merge of `src` into `dst`. Dict values recurse;
    everything else (including lists) overwrites."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _snapshot() -> dict[str, Any]:
    """Deep copy of the current state — safe to hand to JSON serializers
    or external callers without worrying about mutation."""
    return copy.deepcopy(pipeline_state)


async def apply_patch(patch: dict[str, Any], *, source: str = "unknown") -> dict[str, Any]:
    """Deep-merge a patch into pipeline_state and broadcast it.

    Returns the patch as broadcast (which is the input patch — we don't
    rewrite it). `source` tags who originated the change ("agent" |
    "ui" | "system") so subscribers can choose to ignore echoes of their
    own edits.
    """
    async with _lock():
        _deep_merge(pipeline_state, patch)

    msg = {"type": "state_patch", "source": source, "patch": patch}
    # Broadcast outside the lock — sending to subscribers shouldn't
    # serialize against state writes.
    for q in list(_subscribers):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            # Skip slow clients; they'll catch up via a fresh GET /api/state
            # the next time they reconnect.
            pass
    return patch


async def replace_state(new_state: dict[str, Any], *, source: str = "system") -> None:
    """Wholesale replacement (used by load_checkpoint when the entire
    pipeline reconfigures at once). Broadcasts the new state as a single
    patch covering every top-level key."""
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
    """Sync read — safe because dict reads are atomic in CPython and the
    callers that matter (agent tools, GET /api/state) just want a
    snapshot, not transactional consistency with concurrent writers."""
    return _snapshot()


def broadcast_event(ev: dict[str, Any]) -> None:
    """Push a custom event to all WS subscribers without mutating
    pipeline_state. Used by tools that emit live progress (training
    ticks, validation ticks, session reset) — the frontend needs these
    in its reactive stores but they don't fit the deep-merge state
    model (lists append rather than overwrite, transient fields aren't
    part of the shared snapshot)."""
    for q in list(_subscribers):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            pass


# ── Subscription ───────────────────────────────────────────────────────


def subscribe(maxsize: int = 256) -> tuple[asyncio.Queue, callable]:
    """Register a new subscriber. Returns (queue, unsubscribe_fn).

    Caller pulls events off `queue` (each is a dict with `type` and
    payload). Call `unsubscribe_fn()` when the subscriber disconnects.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _subscribers.append(q)

    def unsubscribe() -> None:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass

    return q, unsubscribe


async def stream_events(queue: asyncio.Queue) -> AsyncIterator[dict[str, Any]]:
    """Convenience: async generator that yields events forever from a
    subscription queue. Cancel the consuming task to stop."""
    while True:
        ev = await queue.get()
        yield ev

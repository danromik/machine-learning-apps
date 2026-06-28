"""The autonomous check-in scheduler — what makes overnight runs token-cheap.

During a long background run, the RL Coach must NOT stream continuously (that
burns tokens and context). Instead this server-side asyncio task wakes the Coach
*briefly* on a hybrid cadence — whenever the trainer raises a milestone
(curriculum promotion, checkpoint, finish) OR at least every `cadence_minutes` —
runs one short `run_turn`, and goes back to sleep. The agent reads status,
optionally tweaks the curriculum/hyperparameters, updates the training report,
and stops.

Because the run is overnight/headless, check-ins are driven entirely server-side
(not by a browser timer), so they fire even with the browser closed. Each
autonomous turn's events are bridged onto `/ws/state` as `agent_event` so an open
chat pane renders them live; the SDK also persists the turn to the coach session
JSONL, so a reopened browser can rehydrate the full transcript.

All autonomous turns resume the SAME coach session id (threaded through from the
user's last turn) so context carries across check-ins, and share an
`AGENT_TURN_LOCK` with the user chat path so the two never interleave on one
session (a user message always wins; the scheduler defers).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import agent_state
from agent_runtime import AGENT_TURN_LOCK, run_turn

_POLL_SECONDS = 15.0


class CheckinScheduler:
    def __init__(
        self,
        get_trainer: Callable[[], Any],
        get_coach_session_id: Callable[[], str | None],
        set_coach_session_id: Callable[[str | None], None],
    ):
        self._get_trainer = get_trainer
        self._get_coach_session_id = get_coach_session_id
        self._set_coach_session_id = set_coach_session_id
        self._task: asyncio.Task | None = None
        self._last_checkin = 0.0
        self._run_seen: str | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="checkin-scheduler")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_POLL_SECONDS)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — never let the scheduler die
                continue

    async def _tick(self) -> None:
        t = self._get_trainer()
        if t is None or not t.is_running():
            return
        # Reset the cadence clock when a new run begins.
        if t.run_id != self._run_seen:
            self._run_seen = t.run_id
            self._last_checkin = time.time()

        milestone = t.consume_milestone()
        cadence_s = max(60.0, float(t._cfg.cadence_minutes) * 60.0) if t._cfg else 1200.0
        due_time = (time.time() - self._last_checkin) >= cadence_s
        if not (milestone or due_time):
            return
        # Defer if a user turn (or a prior check-in) is in flight — user wins.
        if AGENT_TURN_LOCK.locked():
            return
        reason = milestone or "scheduled"
        await self._checkin(reason)

    async def _checkin(self, reason: str) -> None:
        t = self._get_trainer()
        if t is None:
            return
        async with AGENT_TURN_LOCK:
            self._last_checkin = time.time()
            status = t.status()
            sid = self._get_coach_session_id()
            agent_state.broadcast_event({
                "type": "agent_event", "source": "coach",
                "event": {"type": "checkin_start", "reason": reason,
                          "run_id": status.get("run_id")},
            })
            prompt = self._prompt(reason, status)
            try:
                async for ev in run_turn(prompt, resume_session_id=sid):
                    agent_state.broadcast_event(
                        {"type": "agent_event", "source": "coach", "event": ev}
                    )
                    if ev.get("type") == "result" and ev.get("session_id"):
                        self._set_coach_session_id(ev["session_id"])
            except Exception as exc:  # noqa: BLE001
                agent_state.broadcast_event({
                    "type": "agent_event", "source": "coach",
                    "event": {"type": "error", "message": f"check-in failed: {exc}"},
                })

    def _prompt(self, reason: str, status: dict) -> str:
        return (
            f"[AUTONOMOUS CHECK-IN — {reason}] You are being woken to review the "
            f"background training run, not by the user. Run {status.get('run_id')}: "
            f"iteration {status.get('iteration')}, current curriculum depth "
            f"k={status.get('current_k')}, solve-rate by depth "
            f"{status.get('solve_rate_by_k')}, last checkpoint "
            f"{status.get('last_checkpoint')}. "
            "Briefly: call get_run_status / get_recent_progress to assess, adjust "
            "the curriculum (set_curriculum_schedule) or hyperparameters only if "
            "it's clearly stuck, then call update_training_report to refresh the "
            "Progress Report tab with a clear, current summary, and stop. Be "
            "concise — this is a routine check-in, not a conversation."
        )

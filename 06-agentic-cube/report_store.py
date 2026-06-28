"""The training report — a live markdown document the RL Coach writes and keeps
current throughout a run, shown in the Progress Report tab (and, when finalized,
the Debrief tab).

Replaces the old pre-packaged status-bucket debrief: instead of a fixed
paragraph chosen by a heuristic, the report is genuine prose the agent authors
from the actual run state, updated on each autonomous check-in.

Stored in memory (for connectionless GET) and mirrored to a file under
`reports/`. Mutations broadcast a `report_update` / `report_final` event over
`/ws/state` so the open tab refreshes live.
"""

from __future__ import annotations

import time
from pathlib import Path

import agent_state

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"

_state: dict = {
    "markdown": "",
    "final": False,
    "updated_at": None,
    "run_id": None,
}


def _path(run_id: str | None, final: bool) -> Path:
    rid = run_id or "current"
    suffix = ".debrief.md" if final else ".md"
    return REPORTS_DIR / f"report_{rid}{suffix}"


def _persist() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _path(_state["run_id"], _state["final"]).write_text(_state["markdown"])
    except OSError:
        pass


def update(markdown: str, *, mode: str = "replace", run_id: str | None = None) -> dict:
    """Replace or append the live report body. `mode` ∈ {"replace", "append"}."""
    if mode == "append" and _state["markdown"]:
        _state["markdown"] = _state["markdown"].rstrip() + "\n\n" + markdown
    else:
        _state["markdown"] = markdown
    _state["final"] = False
    _state["updated_at"] = time.time()
    if run_id is not None:
        _state["run_id"] = run_id
    _persist()
    agent_state.broadcast_event({
        "type": "report_update", "source": "coach",
        "markdown": _state["markdown"], "updated_at": _state["updated_at"],
    })
    return get()


def generate_final(markdown: str, *, run_id: str | None = None) -> dict:
    """Set the final debrief report and broadcast it (the Debrief tab listens)."""
    _state["markdown"] = markdown
    _state["final"] = True
    _state["updated_at"] = time.time()
    if run_id is not None:
        _state["run_id"] = run_id
    _persist()
    agent_state.broadcast_event({
        "type": "report_final", "source": "coach",
        "markdown": _state["markdown"], "updated_at": _state["updated_at"],
    })
    return get()


def reset() -> None:
    _state.update({"markdown": "", "final": False, "updated_at": None, "run_id": None})


def get() -> dict:
    return dict(_state)

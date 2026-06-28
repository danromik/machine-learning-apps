"""MCP tool surface exposed to the RL Coach agent.

These wrap the same backend the user drives through the UI: configure the
environment + reward, pick the algorithm + hyperparameters, build and train a
session, watch the agent play, manage checkpoints, switch device. The agent
reads `agent_state.pipeline_state` and writes via `apply_patch` so every
mutation is broadcast to the browser, keeping the UI in lockstep.

Long-running training is bounded: `train_n_episodes` runs at most 200 episodes
per call and streams an `episode_tick` per episode, giving the agent natural
narration points and keeping the chat responsive.

server.py calls `set_session_accessors(...)` once at startup to bind the live
SnakeSession getter/setter and the device getter/setter (breaks the otherwise
circular import).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import torch
from claude_agent_sdk import create_sdk_mcp_server, tool

import training as _training
from agent_state import apply_patch, broadcast_event, get_state, pipeline_state, replace_state
from agents import ALGORITHMS, HyperParams, default_hyperparams

MCP_SERVER_NAME = "agentic-snake"

# ── Session accessors (injected by server.py) ──────────────────────────
_get_session: Callable[[], _training.SnakeSession | None] | None = None
_set_session: Callable[[_training.SnakeSession | None], None] | None = None
_get_device: Callable[[], torch.device] | None = None
_set_device: Callable[[str], torch.device] | None = None


def set_session_accessors(*, get_session, set_session, get_device, set_device) -> None:
    global _get_session, _set_session, _get_device, _set_device
    _get_session = get_session
    _set_session = set_session
    _get_device = get_device
    _set_device = set_device


# ── Result helpers ─────────────────────────────────────────────────────
def _ok(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"ERROR: {message}"}], "is_error": True}


def _require_session() -> _training.SnakeSession:
    if _get_session is None:
        raise RuntimeError("session accessor not initialized")
    s = _get_session()
    if s is None:
        raise ValueError("no training session — call init_session first")
    return s


def _session_info(s: _training.SnakeSession | None) -> dict:
    if s is None:
        return {"has_session": False}
    recent = s.score_history[-20:]
    return {
        "has_session": True,
        "algo": s.algo,
        "uses_network": s.uses_network(),
        "param_count": s.param_count(),
        "episode": s.episode,
        "best_score": s.best_score,
        "device": str(s.device),
        "recent_mean_score": (
            sum(r["score"] for r in recent) / len(recent) if recent else 0.0
        ),
    }


# ── Reads ──────────────────────────────────────────────────────────────
@tool(
    "get_pipeline_state",
    "Snapshot of the whole pipeline: environment (grid + reward), algorithm "
    "+ hyperparameters, training prefs, and live session status (episodes "
    "trained, best score, recent mean score, parameter count).",
    {},
)
async def get_pipeline_state(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"pipeline": get_state(),
                "session": _session_info(_get_session() if _get_session else None)})


@tool(
    "get_recent_progress",
    "The last N per-episode records (score, reward, length, epsilon/loss). "
    "Use this to judge whether the agent is improving before deciding to "
    "train more, change hyperparameters, or switch algorithm.",
    {"n": int},
)
async def get_recent_progress(args: dict[str, Any]) -> dict[str, Any]:
    s = _get_session() if _get_session else None
    if s is None:
        return _ok({"records": []})
    n = max(1, min(500, int(args.get("n", 50))))
    return _ok({"records": s.score_history[-n:]})


@tool(
    "list_algorithms",
    "The available RL algorithms (qlearning, dqn, reinforce) with a "
    "description of each and its default hyperparameters.",
    {},
)
async def list_algorithms(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"algorithms": {
        k: {**v, "default_hyperparameters": default_hyperparams(k)}
        for k, v in ALGORITHMS.items()
    }})


@tool("list_checkpoints", "All saved .pt checkpoints with size and mtime.", {})
async def list_checkpoints(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"files": _training.list_checkpoints()})


@tool(
    "list_devices",
    "Compute devices available (cpu, mps on Apple Silicon, cuda on NVIDIA) "
    "and which is selected. Note: for these tiny networks CPU is often as "
    "fast or faster than MPS — lots of small per-step forward passes.",
    {},
)
async def list_devices(args: dict[str, Any]) -> dict[str, Any]:
    if _get_device is None:
        return _err("device accessor not initialized")
    devs = [{"name": "cpu", "available": True}]
    if torch.backends.mps.is_available():
        devs.append({"name": "mps", "available": True})
    if torch.cuda.is_available():
        devs.append({"name": "cuda", "available": True})
    return _ok({"current": str(_get_device()), "devices": devs})


# ── Configuration mutations ────────────────────────────────────────────
@tool(
    "set_environment",
    "Configure the Snake environment. `width`/`height` are the grid size "
    "(square grids 6–20 work well). `observation` chooses what the agent "
    "sees: 'features' (an 11-value engineered vector — danger x3, heading x4, "
    "food-direction x4; tiny and fast but blind to the snake's own body) or "
    "'grid' (the full board as a tensor — the agent sees its whole body, but "
    "tabular Q-learning is no longer valid and the deep agents need more "
    "training). `reward` is a dict of components: food (eating, default +1), "
    "death (collision, default -1), step (per-step living cost — negative "
    "discourages stalling), toward_food / away_from_food (shaping for moving "
    "closer/farther; small values like ±0.05 can speed up learning). Pass "
    "only the fields you want to change. Changing observation requires "
    "init_session to rebuild the agent (its network shape changes).",
    {"width": int, "height": int, "observation": str, "reward": dict},
)
async def set_environment(args: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if "width" in args and args["width"] is not None:
        patch["width"] = max(5, min(40, int(args["width"])))
    if "height" in args and args["height"] is not None:
        patch["height"] = max(5, min(40, int(args["height"])))
    if "observation" in args and args["observation"] is not None:
        obs = str(args["observation"]).strip().lower()
        if obs not in ("features", "grid"):
            return _err(f"observation must be 'features' or 'grid'; got {obs!r}")
        patch["observation"] = obs
    if isinstance(args.get("reward"), dict):
        valid = {"food", "death", "step", "toward_food", "away_from_food"}
        bad = set(args["reward"]) - valid
        if bad:
            return _err(f"unknown reward keys: {sorted(bad)}; valid: {sorted(valid)}")
        patch["reward"] = {k: float(v) for k, v in args["reward"].items()}
    if not patch:
        return _err("nothing to change — pass width, height, observation, and/or reward")
    await apply_patch({"environment": patch}, source="agent")
    return _ok({"environment": get_state()["environment"]})


@tool(
    "set_algorithm",
    "Choose the RL algorithm: 'qlearning' (tabular, no network), 'dqn' "
    "(deep value-based), or 'reinforce' (policy gradient). Resets the "
    "hyperparameters to that algorithm's defaults. Call init_session "
    "afterward to build a fresh agent.",
    {"algo": str},
)
async def set_algorithm(args: dict[str, Any]) -> dict[str, Any]:
    algo = str(args.get("algo", "")).strip().lower()
    if algo not in ALGORITHMS:
        return _err(f"unknown algorithm {algo!r}; valid: {sorted(ALGORITHMS)}")
    await apply_patch(
        {"algorithm": {"algo": algo, "hyperparameters": default_hyperparams(algo)}},
        source="agent",
    )
    return _ok({"algo": algo, "hyperparameters": default_hyperparams(algo)})


@tool(
    "set_hyperparameters",
    "Update training hyperparameters (merged into the current set). Common "
    "knobs: lr (learning rate), gamma (discount 0–1), epsilon_start / "
    "epsilon_min / epsilon_decay (exploration schedule, value-based only), "
    "hidden (MLP width), batch_size / buffer_size / target_update (DQN). If "
    "a session is live, lr is hot-swapped on the next train call.",
    {"hyperparameters": dict},
)
async def set_hyperparameters(args: dict[str, Any]) -> dict[str, Any]:
    hp = args.get("hyperparameters")
    if not isinstance(hp, dict) or not hp:
        return _err("hyperparameters must be a non-empty dict")
    valid = set(HyperParams().__dict__.keys())
    bad = set(hp) - valid
    if bad:
        return _err(f"unknown hyperparameter keys: {sorted(bad)}; valid: {sorted(valid)}")
    await apply_patch({"algorithm": {"hyperparameters": hp}}, source="agent")
    s = _get_session() if _get_session else None
    if s is not None:
        s.update_hyperparameters(hp)
    return _ok({"hyperparameters": get_state()["algorithm"]["hyperparameters"]})


# ── Training operations ────────────────────────────────────────────────
@tool(
    "init_session",
    "Build a fresh agent + environment from the current pipeline state "
    "(algorithm + hyperparameters + environment). Replaces any existing "
    "session — all learned progress is discarded. Call this before training "
    "a new configuration.",
    {},
)
async def init_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None or _get_device is None:
        return _err("session accessors not initialized")
    state = get_state()
    algo = state["algorithm"]["algo"]
    if algo not in ALGORITHMS:
        return _err(f"unknown algorithm: {algo}")
    try:
        session = _training.SnakeSession(
            algo, state["algorithm"]["hyperparameters"], state["environment"], _get_device(),
        )
    except ValueError as e:
        # e.g. tabular Q-learning requested on the grid observation.
        return _err(str(e))
    _set_session(session)
    broadcast_event({
        "type": "training_session", "source": "agent", "hasSession": True,
        "summary": _session_info(session), "scoreHistory": [],
    })
    return _ok(_session_info(session))


@tool(
    "reset_session",
    "Drop the live session — discards the agent's learned weights/table and "
    "clears the score chart. Use to scrap progress and start over without "
    "changing the configuration.",
    {},
)
async def reset_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None:
        return _err("session accessors not initialized")
    _set_session(None)
    broadcast_event({"type": "training_session", "source": "agent", "hasSession": False})
    return _ok({"has_session": False})


@tool(
    "train_n_episodes",
    "Train for N episodes (full games), learning from each. Streams a live "
    "tick per episode to the UI's score chart. Returns mean/best/last score "
    "across the run. Bounded to 200 per call — call again to keep training. "
    "Watch the mean score climb to judge learning.",
    {"n": int},
)
async def train_n_episodes(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    n = max(1, min(200, int(args.get("n", 1))))
    scores: list[int] = []
    for _ in range(n):
        record = s.train_one_episode()
        scores.append(record["score"])
        broadcast_event({"type": "episode_tick", "source": "agent", "record": record})
        # Yield so the WS sender flushes the tick before the next episode.
        await asyncio.sleep(0)
    return _ok({
        "episodes_run": n,
        "episode": s.episode,
        "mean_score": sum(scores) / len(scores),
        "best_score": s.best_score,
        "last_score": scores[-1],
        "recent_metrics": s.score_history[-1] if s.score_history else None,
    })


@tool(
    "evaluate",
    "Greedy evaluation over N episodes with exploration turned off — this is "
    "the policy you'd actually deploy. Returns mean/best score and mean "
    "episode length. Doesn't change the agent.",
    {"n": int},
)
async def evaluate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    return _ok(s.evaluate(max(1, min(200, int(args.get("n", 20))))))


@tool(
    "watch_agent_play",
    "Run one greedy game and return a summary of what happened: final score, "
    "length, and how it ended (death / timeout / win). Use this to diagnose "
    "behavior — e.g. 'survives but won't chase food' or 'traps itself at "
    "length 20'. (The UI's Watch tab animates the full game separately.)",
    {},
)
async def watch_agent_play(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    result = s.play_episode(greedy=True)
    last_event = result["steps"][-1]["event"] if result["steps"] else "none"
    return _ok({
        "score": result["score"],
        "length": result["length"],
        "ended_with": last_event,
    })


# ── Checkpoints ────────────────────────────────────────────────────────
@tool(
    "save_checkpoint",
    "Save the live session to a .pt file under checkpoints/ — agent "
    "weights/table, algorithm, hyperparameters, environment, and score "
    "history. Filename is sandboxed (no path separators or '..').",
    {"filename": str},
)
async def save_checkpoint(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    filename = str(args.get("filename", "")).strip()
    if not filename:
        return _err("filename is required")
    try:
        name = _training.save_checkpoint(s, filename)
    except ValueError as e:
        return _err(str(e))
    return _ok({"name": name})


@tool(
    "load_checkpoint",
    "Load a checkpoint by filename. Replaces the live session AND restores "
    "the environment + algorithm + hyperparameters in the UI so the agent "
    "and user see the configuration the model was trained against.",
    {"filename": str},
)
async def load_checkpoint(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None or _get_device is None:
        return _err("session accessors not initialized")
    filename = str(args.get("filename", "")).strip()
    if not filename:
        return _err("filename is required")
    try:
        session = _training.load_checkpoint(filename, _get_device())
    except FileNotFoundError:
        return _err(f"no such checkpoint: {filename}")
    _set_session(session)
    from dataclasses import asdict
    new_state = get_state()
    new_state["algorithm"] = {"algo": session.algo, "hyperparameters": asdict(session.hp)}
    new_state["environment"] = _training.env_config_to_dict(session.env_cfg)
    await replace_state(new_state, source="agent")
    broadcast_event({
        "type": "training_session", "source": "agent", "hasSession": True,
        "summary": _session_info(session), "scoreHistory": session.score_history,
    })
    return _ok({**_session_info(session), "name": filename})


@tool(
    "delete_checkpoint",
    "Remove a checkpoint file. Filename sandboxed to checkpoints/.",
    {"filename": str},
)
async def delete_checkpoint(args: dict[str, Any]) -> dict[str, Any]:
    filename = str(args.get("filename", "")).strip()
    if not filename:
        return _err("filename is required")
    try:
        name = _training.delete_checkpoint(filename)
    except FileNotFoundError:
        return _err(f"no such checkpoint: {filename}")
    except ValueError as e:
        return _err(str(e))
    return _ok({"name": name, "deleted": True})


# ── Device ─────────────────────────────────────────────────────────────
@tool(
    "select_device",
    "Switch compute device to 'cpu', 'mps' (Apple GPU), or 'cuda'. A live "
    "session's tensors are migrated. For these tiny nets CPU is usually fine.",
    {"name": str},
)
async def select_device(args: dict[str, Any]) -> dict[str, Any]:
    if _set_device is None:
        return _err("device accessor not initialized")
    name = str(args.get("name", "")).strip().lower()
    if name not in ("cpu", "mps", "cuda"):
        return _err(f"name must be cpu/mps/cuda; got {name!r}")
    try:
        new_device = _set_device(name)
    except ValueError as e:
        return _err(str(e))
    return _ok({"current": str(new_device)})


# ── Build the MCP server ───────────────────────────────────────────────
ALL_TOOLS = [
    get_pipeline_state, get_recent_progress, list_algorithms, list_checkpoints, list_devices,
    set_environment, set_algorithm, set_hyperparameters,
    init_session, reset_session, train_n_episodes, evaluate, watch_agent_play,
    save_checkpoint, load_checkpoint, delete_checkpoint,
    select_device,
]


def build_mcp_server():
    return create_sdk_mcp_server(name=MCP_SERVER_NAME, version="0.1.0", tools=ALL_TOOLS)


def allowed_tool_names() -> list[str]:
    return [f"mcp__{MCP_SERVER_NAME}__{t.name if hasattr(t, 'name') else t.__name__}"
            for t in ALL_TOOLS]

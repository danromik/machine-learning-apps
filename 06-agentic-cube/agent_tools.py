"""MCP tool surface exposed to the RL Coach agent.

These wrap the same backend the user drives through the UI: configure the cube +
curriculum, set hyperparameters, build/train a session, evaluate, manage the
long-running background training run, write the training report, and manage
checkpoints. The agent reads `agent_state.pipeline_state` and writes via
`apply_patch` so every mutation is broadcast to the browser, keeping the UI in
lockstep.

Two kinds of training:
  * `train_n_iterations` — short, foreground, agent-driven chunks (≤200), for
    hands-on experimentation. Streams a `trainer_progress` tick per iteration.
  * `start_training_run` / `stop_training_run` — the **background** run: a daemon
    thread that trains overnight, advances the curriculum, and checkpoints
    itself. The agent kicks it off and then checks in periodically (the check-in
    scheduler wakes it) rather than babysitting — that's what keeps token use
    modest. `get_run_status` is the cheap per-check-in read.

Session mutations go through the shared `SESSION_LOCK` so a check-in can safely
read/tweak the session while the background thread trains between chunks.

server.py calls `set_session_accessors(...)` once at startup to bind the live
CubeSession getter/setter, the device getter/setter, and the background trainer
(breaks the otherwise circular import).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import torch
from claude_agent_sdk import create_sdk_mcp_server, tool

import report_store
import training as _training
from agent_state import apply_patch, broadcast_event, get_state, replace_state
from agents import ALGORITHMS, HyperParams, default_hyperparams
from background_trainer import SESSION_LOCK, RunConfig

MCP_SERVER_NAME = "agentic-cube"

# ── Accessors (injected by server.py) ──────────────────────────────────
_get_session: Callable[[], _training.CubeSession | None] | None = None
_set_session: Callable[[_training.CubeSession | None], None] | None = None
_get_device: Callable[[], torch.device] | None = None
_set_device: Callable[[str], torch.device] | None = None
_get_trainer: Callable[[], Any] | None = None


def set_session_accessors(*, get_session, set_session, get_device, set_device, get_trainer) -> None:
    global _get_session, _set_session, _get_device, _set_device, _get_trainer
    _get_session = get_session
    _set_session = set_session
    _get_device = get_device
    _set_device = set_device
    _get_trainer = get_trainer


# ── Result helpers ─────────────────────────────────────────────────────
def _ok(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"ERROR: {message}"}], "is_error": True}


def _require_session() -> _training.CubeSession:
    if _get_session is None:
        raise RuntimeError("session accessor not initialized")
    s = _get_session()
    if s is None:
        raise ValueError("no training session — call init_session or start_training_run first")
    return s


def _trainer():
    return _get_trainer() if _get_trainer else None


def _session_info(s: _training.CubeSession | None) -> dict:
    if s is None:
        return {"has_session": False}
    return {
        "has_session": True,
        "algo": s.algo,
        "cube_size": s.size,
        "param_count": s.param_count(),
        "iteration": s.iteration,
        "current_k": s.current_k,
        "solve_rate_by_k": dict(s.solve_rate_by_k),
        "device": str(s.device),
        "last_loss": s.loss_history[-1]["loss"] if s.loss_history else None,
    }


def _build_session_from_state() -> _training.CubeSession:
    state = get_state()
    algo = state["algorithm"]["algo"]
    session = _training.CubeSession(
        algo, state["algorithm"]["hyperparameters"],
        {"size": state["environment"]["size"]}, _get_device(),
    )
    cur = state["environment"].get("curriculum", {})
    session.current_k = int(cur.get("startK", 1))
    return session


# ── Reads ──────────────────────────────────────────────────────────────
@tool(
    "get_pipeline_state",
    "Snapshot of the whole pipeline: environment (cube size + reverse-scramble "
    "curriculum), algorithm + hyperparameters, training prefs, and live session "
    "status (iterations trained, current curriculum depth k, solve-rate by depth, "
    "parameter count).",
    {},
)
async def get_pipeline_state(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"pipeline": get_state(),
                "session": _session_info(_get_session() if _get_session else None)})


@tool(
    "get_recent_progress",
    "The last N per-iteration training records (loss, mean_target, curriculum "
    "depth k). Use this to judge whether the cost-to-go function is converging "
    "before deciding to train more or tweak hyperparameters.",
    {"n": int},
)
async def get_recent_progress(args: dict[str, Any]) -> dict[str, Any]:
    s = _get_session() if _get_session else None
    if s is None:
        return _ok({"records": []})
    n = max(1, min(500, int(args.get("n", 50))))
    return _ok({"records": s.loss_history[-n:]})


@tool(
    "get_run_status",
    "Status of the background training run: state (idle/running/stopped/finished/"
    "error), run id, iterations done, current curriculum depth k, solve-rate by "
    "depth, last checkpoint. This is the cheap read to call on each check-in to "
    "decide whether to adjust the curriculum, update the report, or let it run.",
    {},
)
async def get_run_status(args: dict[str, Any]) -> dict[str, Any]:
    t = _trainer()
    if t is None:
        return _err("trainer not initialized")
    return _ok(t.status())


@tool(
    "list_algorithms",
    "The available algorithms (currently value_iteration) with a description "
    "and default hyperparameters.",
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
    "Compute devices available (cpu, mps on Apple Silicon, cuda on NVIDIA) and "
    "which is selected. For value iteration the bottleneck is generating and "
    "expanding scrambled states, so MPS helps less than for big CNNs — try both.",
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
    "Configure the cube + reverse-scramble curriculum. `size` is 2 (Pocket "
    "Cube, ~3.6M states, fully solvable fast) or 3 (standard, ~4.3e19 states — "
    "best-effort overnight). `curriculum` is a dict with startK (depth to begin "
    "at), maxK (deepest scramble to ramp to), and promoteAt (solve-rate at the "
    "current depth that triggers promotion to the next, e.g. 0.9). Pass only the "
    "fields you want to change. Changing `size` requires init_session (the "
    "network's input dimension changes).",
    {"size": int, "curriculum": dict},
)
async def set_environment(args: dict[str, Any]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if args.get("size") is not None:
        size = int(args["size"])
        if size not in (2, 3):
            return _err(f"cube size must be 2 or 3; got {size}")
        patch["size"] = size
    if isinstance(args.get("curriculum"), dict):
        valid = {"startK", "maxK", "promoteAt"}
        bad = set(args["curriculum"]) - valid
        if bad:
            return _err(f"unknown curriculum keys: {sorted(bad)}; valid: {sorted(valid)}")
        cur = {}
        for key in ("startK", "maxK"):
            if key in args["curriculum"]:
                cur[key] = max(1, int(args["curriculum"][key]))
        if "promoteAt" in args["curriculum"]:
            cur["promoteAt"] = float(args["curriculum"]["promoteAt"])
        patch["curriculum"] = cur
    if not patch:
        return _err("nothing to change — pass size and/or curriculum")
    await apply_patch({"environment": patch}, source="agent")
    return _ok({"environment": get_state()["environment"]})


@tool(
    "set_algorithm",
    "Choose the learning algorithm. Currently only 'value_iteration' (cost-to-go "
    "regression with the cube's known model). Resets hyperparameters to defaults.",
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
    "Update training hyperparameters (merged into the current set). Knobs: lr "
    "(learning rate), batch_size (scrambled states per gradient step), hidden "
    "(MLP width), target_update (batches between target-network syncs), "
    "weight_decay. If a session is live, lr is hot-swapped immediately.",
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
    with SESSION_LOCK:
        s = _get_session() if _get_session else None
        if s is not None:
            s.update_hyperparameters(hp)
    return _ok({"hyperparameters": get_state()["algorithm"]["hyperparameters"]})


# ── Training operations ────────────────────────────────────────────────
@tool(
    "init_session",
    "Build a fresh cost-to-go network from the current pipeline state (algorithm "
    "+ hyperparameters + cube size). Replaces any existing session — all learned "
    "progress is discarded. Call before training a new configuration.",
    {},
)
async def init_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None or _get_device is None:
        return _err("session accessors not initialized")
    try:
        with SESSION_LOCK:
            session = _build_session_from_state()
            _set_session(session)
    except ValueError as e:
        return _err(str(e))
    broadcast_event({
        "type": "training_session", "source": "agent", "hasSession": True,
        "summary": _session_info(session), "lossHistory": [],
    })
    return _ok(_session_info(session))


@tool(
    "reset_session",
    "Drop the live session — discards the network weights and clears the loss "
    "chart. Use to scrap progress and start over without changing config.",
    {},
)
async def reset_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None:
        return _err("session accessors not initialized")
    with SESSION_LOCK:
        _set_session(None)
    broadcast_event({"type": "training_session", "source": "agent", "hasSession": False})
    return _ok({"has_session": False})


@tool(
    "train_n_iterations",
    "Foreground training: run N value-iteration gradient steps at the current "
    "curriculum depth, learning from each. Streams a live tick per iteration to "
    "the loss chart. Bounded to 200 per call. Use this for quick hands-on "
    "experiments; for long/overnight training use start_training_run instead. "
    "Errors if a background run is active (it already owns the session).",
    {"n": int},
)
async def train_n_iterations(args: dict[str, Any]) -> dict[str, Any]:
    t = _trainer()
    if t is not None and t.is_running():
        return _err("a background training run is active — use get_run_status, or stop it first")
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    n = max(1, min(200, int(args.get("n", 1))))
    losses: list[float] = []
    for _ in range(n):
        with SESSION_LOCK:
            record = s.train_one_iteration()
        losses.append(record["loss"])
        broadcast_event({
            "type": "trainer_progress", "source": "agent", "record": record,
            "iteration": s.iteration, "current_k": s.current_k,
            "solve_rate_by_k": dict(s.solve_rate_by_k),
        })
        await asyncio.sleep(0)  # let the WS sender flush the tick
    return _ok({
        "iterations_run": n, "iteration": s.iteration, "current_k": s.current_k,
        "mean_loss": sum(losses) / len(losses), "last_loss": losses[-1],
    })


@tool(
    "set_curriculum_depth",
    "Set the live session's current scramble depth k — the depth foreground "
    "training, evaluation, and Watch use. The normal way the curriculum deepens "
    "is automatic (a background run promotes k as each depth is mastered), but "
    "use this to manually jump the depth: e.g. to probe how the solver does at a "
    "harder depth, or to back off to an easier one. Clamped to 1..maxK.",
    {"k": int},
)
async def set_curriculum_depth(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    if args.get("k") is None:
        return _err("k is required")
    max_k = int(get_state()["environment"].get("curriculum", {}).get("maxK", 14))
    k = max(1, min(max_k, int(args["k"])))
    with SESSION_LOCK:
        s.current_k = k
    # Reflect the change in the UI without disturbing the loss chart.
    broadcast_event({
        "type": "training_session", "source": "agent", "hasSession": True,
        "summary": _session_info(s),
    })
    return _ok({"current_k": k})


@tool(
    "evaluate",
    "Measure solve-rate: attempt N freshly-scrambled cubes at depth k (defaults "
    "to the current curriculum depth) using the learned heuristic + beam search. "
    "Returns the fraction solved and mean solution length. This is the real "
    "metric of whether the cube is being solved.",
    {"n": int, "k": int},
)
async def evaluate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    n = max(1, min(300, int(args.get("n", 50))))
    k = int(args["k"]) if args.get("k") is not None else None
    with SESSION_LOCK:
        return _ok(s.evaluate(n=n, k=k))


@tool(
    "watch_agent_play",
    "Scramble a cube k moves and try to solve it, returning a summary: whether "
    "it was solved, the solution length, and the scramble depth. Use to diagnose "
    "where the solver breaks down. (The UI's Watch tab animates the full 3D solve "
    "separately.)",
    {"k": int},
)
async def watch_agent_play(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    k = int(args["k"]) if args.get("k") is not None else None
    with SESSION_LOCK:
        result = s.solve_episode(k=k)
    return _ok({
        "solved": result["solved"],
        "scramble_depth": result["scramble_depth"],
        "solution_len": result["solution_len"],
    })


# ── Background run control ─────────────────────────────────────────────
@tool(
    "start_training_run",
    "Start the BACKGROUND training run — a long, overnight-capable job that "
    "trains independently of this chat, advances the reverse-scramble curriculum "
    "automatically, and checkpoints itself periodically. Builds a fresh session "
    "from the current config. Args (all optional, default from pipeline state): "
    "start_k, max_k, promote_at, max_iterations (0 = until stopped), eval_every, "
    "checkpoint_every, cadence_minutes (how often you'll be woken to check in). "
    "After starting, DON'T babysit — you'll be woken on milestones to review and "
    "update the report.",
    {"start_k": int, "max_k": int, "promote_at": float, "max_iterations": int,
     "eval_every": int, "checkpoint_every": int, "cadence_minutes": float},
)
async def start_training_run(args: dict[str, Any]) -> dict[str, Any]:
    t = _trainer()
    if t is None:
        return _err("trainer not initialized")
    if t.is_running():
        return _err("a training run is already in progress — stop it first")
    state = get_state()
    env = state["environment"]
    cur = env.get("curriculum", {})
    tr = state["training"]
    cfg = RunConfig(
        cube_size=int(env["size"]),
        start_k=int(args.get("start_k", cur.get("startK", 1))),
        max_k=int(args.get("max_k", cur.get("maxK", 14))),
        promote_at=float(args.get("promote_at", cur.get("promoteAt", 0.9))),
        max_iterations=int(args.get("max_iterations", 0)),
        eval_every=int(args.get("eval_every", tr.get("evalEveryN", 100))),
        eval_n=int(tr.get("evalN", 80)),
        checkpoint_every=int(args.get("checkpoint_every", 200)),
        cadence_minutes=float(args.get("cadence_minutes", tr.get("cadenceMinutes", 20))),
    )
    try:
        status = t.start(cfg, fresh=True)
    except (RuntimeError, ValueError) as e:
        return _err(str(e))
    return _ok({"started": True, "status": status})


@tool(
    "stop_training_run",
    "Stop the background training run. Writes a final checkpoint first. The "
    "learned model stays loaded as the live session so you can evaluate it.",
    {},
)
async def stop_training_run(args: dict[str, Any]) -> dict[str, Any]:
    t = _trainer()
    if t is None:
        return _err("trainer not initialized")
    return _ok({"stopped": True, "status": t.stop()})


@tool(
    "set_curriculum_schedule",
    "Adjust the curriculum of a RUNNING background run on the fly: max_k "
    "(deepest scramble) and/or promote_at (the solve-rate bar for promotion). "
    "Use this from a check-in if the curriculum is stuck (e.g. lower promote_at "
    "to keep progressing, or raise max_k to push deeper). Also updates the UI.",
    {"max_k": int, "promote_at": float},
)
async def set_curriculum_schedule(args: dict[str, Any]) -> dict[str, Any]:
    t = _trainer()
    patch: dict[str, Any] = {}
    if args.get("max_k") is not None:
        patch["maxK"] = max(1, int(args["max_k"]))
    if args.get("promote_at") is not None:
        patch["promoteAt"] = float(args["promote_at"])
    if not patch:
        return _err("pass max_k and/or promote_at")
    await apply_patch({"environment": {"curriculum": patch}}, source="agent")
    if t is not None and t.is_running():
        t.update_curriculum(max_k=patch.get("maxK"), promote_at=patch.get("promoteAt"))
    return _ok({"curriculum": get_state()["environment"]["curriculum"],
                "applied_to_run": bool(t and t.is_running())})


# ── Training report ────────────────────────────────────────────────────
@tool(
    "update_training_report",
    "Write or update the live training report shown in the Progress Report tab "
    "(markdown). This is how the user follows a long run without reading the "
    "chat. Keep it genuinely informative: current curriculum depth and solve-"
    "rate, what's working, what you changed and why, and what to expect next. "
    "mode='replace' overwrites the whole report (preferred); 'append' adds a "
    "timestamped note.",
    {"markdown": str, "mode": str},
)
async def update_training_report(args: dict[str, Any]) -> dict[str, Any]:
    md = str(args.get("markdown", "")).strip()
    if not md:
        return _err("markdown is required")
    mode = str(args.get("mode", "replace")).strip().lower()
    if mode not in ("replace", "append"):
        mode = "replace"
    t = _trainer()
    run_id = t.run_id if t else None
    report_store.update(md, mode=mode, run_id=run_id)
    return _ok({"updated": True, "mode": mode})


@tool(
    "generate_final_report",
    "Write the FINAL debrief report (markdown), shown in the Debrief tab. Call "
    "this when training is finished (the user clicked Finish Training, or the run "
    "completed). Summarize the whole run: what was achieved (solve-rate by depth, "
    "whether the cube is genuinely solvable now), the key decisions and what they "
    "taught, the limits hit, and honest next steps.",
    {"markdown": str},
)
async def generate_final_report(args: dict[str, Any]) -> dict[str, Any]:
    md = str(args.get("markdown", "")).strip()
    if not md:
        return _err("markdown is required")
    t = _trainer()
    run_id = t.run_id if t else None
    report_store.generate_final(md, run_id=run_id)
    return _ok({"finalized": True})


# ── Checkpoints ────────────────────────────────────────────────────────
@tool(
    "save_checkpoint",
    "Save the live session to a .pt file under checkpoints/ — network weights, "
    "algorithm, hyperparameters, cube config, curriculum depth, and loss "
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
        with SESSION_LOCK:
            name = _training.save_checkpoint(s, filename)
    except ValueError as e:
        return _err(str(e))
    return _ok({"name": name})


@tool(
    "load_checkpoint",
    "Load a checkpoint by filename. Replaces the live session AND restores the "
    "cube size + algorithm + hyperparameters in the UI so the agent and user see "
    "the configuration the model was trained against.",
    {"filename": str},
)
async def load_checkpoint(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None or _get_device is None:
        return _err("session accessors not initialized")
    t = _trainer()
    if t is not None and t.is_running():
        return _err("stop the background run before loading a checkpoint")
    filename = str(args.get("filename", "")).strip()
    if not filename:
        return _err("filename is required")
    try:
        with SESSION_LOCK:
            session = _training.load_checkpoint(filename, _get_device())
            _set_session(session)
    except FileNotFoundError:
        return _err(f"no such checkpoint: {filename}")
    from dataclasses import asdict
    new_state = get_state()
    new_state["algorithm"] = {"algo": session.algo, "hyperparameters": asdict(session.hp)}
    new_state["environment"]["size"] = session.size
    await replace_state(new_state, source="agent")
    broadcast_event({
        "type": "training_session", "source": "agent", "hasSession": True,
        "summary": _session_info(session), "lossHistory": session.loss_history,
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
    "session's tensors are migrated.",
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
    get_pipeline_state, get_recent_progress, get_run_status,
    list_algorithms, list_checkpoints, list_devices,
    set_environment, set_algorithm, set_hyperparameters,
    init_session, reset_session, train_n_iterations, set_curriculum_depth,
    evaluate, watch_agent_play,
    start_training_run, stop_training_run, set_curriculum_schedule,
    update_training_report, generate_final_report,
    save_checkpoint, load_checkpoint, delete_checkpoint,
    select_device,
]


def build_mcp_server():
    return create_sdk_mcp_server(name=MCP_SERVER_NAME, version="0.1.0", tools=ALL_TOOLS)


def allowed_tool_names() -> list[str]:
    return [f"mcp__{MCP_SERVER_NAME}__{t.name if hasattr(t, 'name') else t.__name__}"
            for t in ALL_TOOLS]

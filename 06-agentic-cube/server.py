"""FastAPI backend for the Agentic Cube Trainer.

Same architecture as the other agentic apps: a shared pipeline-state mirror that
both the UI and the RL Coach agent drive, a /ws/state socket that broadcasts
state patches + live training progress, and an SSE agent-chat endpoint. The
domain underneath is reinforcement learning on a Rubik's Cube (a CubeSession of
cube + cost-to-go agent), with a **background training run** for overnight
training and a server-side **check-in scheduler** that wakes the RL Coach
periodically.

Endpoints
  GET  /api/device  ·  GET /api/device/list  ·  POST /api/device/select
  GET  /api/catalog                         algorithms + their default hyperparameters

  GET  /api/state  ·  POST /api/state/patch  ·  WS /ws/state

  POST /api/training/init                   build a session from the state mirror
  GET  /api/training/state
  POST /api/training/reset
  POST /api/training/train_iterations       run a bounded chunk of value-iteration steps
  POST /api/training/eval                   solve-rate evaluation (no learning)
  POST /api/training/play                   one 3D solve, with frames + per-step scores
  GET  /api/training/checkpoints  ·  save  ·  load  ·  delete

  POST /api/training/run/start  ·  /stop  ·  GET /status  ·  POST /resume
  GET  /api/training/report                 the live (or final) RL-Coach report

  POST /api/agent/chat (SSE)  ·  POST /api/agent/stop
  GET  /api/agent/sessions  ·  GET /api/agent/sessions/{id}/messages
"""
from __future__ import annotations

import asyncio
import json
import platform
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

import agent_runtime as _agent_runtime  # noqa: E402
import agent_state as _agent_state  # noqa: E402
import agent_tools as _agent_tools  # noqa: E402
import background_trainer as _bg  # noqa: E402
import report_store as _report_store  # noqa: E402
import training as _training  # noqa: E402
from agents import ALGORITHMS, default_hyperparams  # noqa: E402
from background_trainer import SESSION_LOCK, BackgroundTrainer, RunConfig  # noqa: E402
from checkin_scheduler import CheckinScheduler  # noqa: E402
from models import pick_device  # noqa: E402

STATIC_DIR = _HERE / "frontend" / "dist"

device = pick_device()

app = FastAPI(title="Agentic Cube Trainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Device info (macOS-aware) ──────────────────────────────────────────
def _sysctl(name: str) -> str | None:
    import subprocess
    try:
        r = subprocess.run(["sysctl", "-n", name], capture_output=True,
                           text=True, timeout=2)
        return r.stdout.strip() or None
    except Exception:
        return None


def _system_memory_bytes() -> int | None:
    import os
    val = _sysctl("hw.memsize")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, OSError):
        pass
    return None


def _device_descriptor(name: str) -> dict:
    info: dict = {"name": name}
    mem = _system_memory_bytes()
    if name == "cpu":
        info["label"] = (_sysctl("machdep.cpu.brand_string")
                         or platform.processor() or platform.machine() or "CPU")
        cores = _sysctl("hw.ncpu")
        if cores:
            try:
                info["cores"] = int(cores)
            except ValueError:
                pass
        info["memory_bytes"] = mem
    elif name == "mps":
        chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
        info["label"] = f"GPU ({chip})"
        info["memory_bytes"] = mem
        info["memory_note"] = "unified with CPU"
    elif name == "cuda":
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["label"] = props.name
            info["memory_bytes"] = props.total_memory
    return info


def _list_devices() -> list[dict]:
    out: list[dict] = [{**_device_descriptor("cpu"), "available": True}]
    if torch.backends.mps.is_available():
        out.append({**_device_descriptor("mps"), "available": True})
    if torch.cuda.is_available():
        out.append({**_device_descriptor("cuda"), "available": True})
    return out


@app.get("/api/device")
def get_device():
    return {"device": str(device)}


@app.get("/api/device/list")
def get_device_list():
    s = _training_state["session"]
    return {
        "current": str(device),
        "devices": _list_devices(),
        "session_loaded": s is not None,
        "param_count": s.param_count() if s is not None else 0,
    }


class DeviceSelectReq(BaseModel):
    name: str


def _select_device_impl(name: str) -> torch.device:
    """Swap the global device and migrate a live session's tensors. Shared by
    the REST endpoint and the agent's select_device tool."""
    global device
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS not available on this system")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available on this system")
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError(f"unknown device: {name!r}")
    new_device = torch.device(name)
    if str(new_device) != str(device):
        device = new_device
        with SESSION_LOCK:
            s = _training_state["session"]
            if s is not None:
                s.move_to_device(device)
    return device


@app.post("/api/device/select")
def post_device_select(req: DeviceSelectReq):
    try:
        return {"current": str(_select_device_impl(req.name))}
    except ValueError as e:
        raise HTTPException(400, str(e))


# ── Catalog ────────────────────────────────────────────────────────────
@app.get("/api/catalog")
def get_catalog():
    algos = []
    for key, meta in ALGORITHMS.items():
        algos.append({"id": key, **meta, "default_hyperparameters": default_hyperparams(key)})
    return {"algorithms": algos}


# ── Training session ───────────────────────────────────────────────────
_training_state: dict = {"session": None}


def _build_session_from_state() -> _training.CubeSession:
    """Construct a CubeSession from the current pipeline-state mirror."""
    state = _agent_state.get_state()
    algo = state["algorithm"]["algo"]
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algo}")
    session = _training.CubeSession(
        algo, state["algorithm"]["hyperparameters"],
        {"size": state["environment"]["size"]}, device,
    )
    cur = state["environment"].get("curriculum", {})
    session.current_k = int(cur.get("startK", 1))
    return session


trainer = BackgroundTrainer(
    get_session=lambda: _training_state["session"],
    set_session=lambda s: _training_state.__setitem__("session", s),
    get_device=lambda: device,
    build_session_from_state=_build_session_from_state,
)

_agent_tools.set_session_accessors(
    get_session=lambda: _training_state["session"],
    set_session=lambda s: _training_state.__setitem__("session", s),
    get_device=lambda: device,
    set_device=_select_device_impl,
    get_trainer=lambda: trainer,
)

# Latest RL Coach session id — threaded through autonomous check-ins so they
# resume the user's coaching session.
_coach_session: dict[str, str | None] = {"id": None}

scheduler = CheckinScheduler(
    get_trainer=lambda: trainer,
    get_coach_session_id=lambda: _coach_session["id"],
    set_coach_session_id=lambda sid: _coach_session.__setitem__("id", sid),
)


def _require_session() -> _training.CubeSession:
    s = _training_state["session"]
    if s is None:
        raise HTTPException(409, "no training session — initialize first")
    return s


def _session_summary(s: _training.CubeSession) -> dict:
    return {
        "has_session": True,
        "algo": s.algo,
        "cube_size": s.size,
        "uses_network": s.uses_network(),
        "param_count": s.param_count(),
        "iteration": s.iteration,
        "current_k": s.current_k,
        "solve_rate_by_k": dict(s.solve_rate_by_k),
        "device": str(s.device),
    }


def _refuse_if_running() -> None:
    if trainer.is_running():
        raise HTTPException(409, "a background training run is active — stop it first")


@app.post("/api/training/init")
def post_training_init():
    """Build a fresh session from the current pipeline-state mirror."""
    _refuse_if_running()
    try:
        with SESSION_LOCK:
            session = _build_session_from_state()
            _training_state["session"] = session
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _session_summary(session)


@app.get("/api/training/state")
def get_training_state():
    s = _training_state["session"]
    if s is None:
        return {"has_session": False}
    return {**_session_summary(s), "loss_history": s.loss_history}


@app.post("/api/training/reset")
def post_training_reset():
    _refuse_if_running()
    _training_state["session"] = None
    return {"has_session": False}


class TrainIterReq(BaseModel):
    n: int = 5
    k: int | None = None
    hyperparameters: dict | None = None


@app.post("/api/training/train_iterations")
def post_train_iterations(req: TrainIterReq):
    """Run a bounded chunk of value-iteration steps and return one record per
    iteration. The frontend calls this in a Stop-able loop to keep the loss
    chart live. (Foreground only — refuses while a background run owns the
    session.)"""
    _refuse_if_running()
    s = _require_session()
    if req.hyperparameters:
        s.update_hyperparameters(req.hyperparameters)
    # Honor a manually-set curriculum depth: persist it on the session so eval /
    # play / the coach's status all reflect the depth the user is training at.
    if req.k is not None:
        s.current_k = max(1, int(req.k))
    n = max(1, min(100, int(req.n)))
    records: list[dict] = []
    summary = s.train_iterations(n, k=s.current_k, on_iteration=records.append)
    return {"records": records, "summary": summary}


class EvalReq(BaseModel):
    n: int = 50
    k: int | None = None


@app.post("/api/training/eval")
def post_eval(req: EvalReq):
    s = _require_session()
    with SESSION_LOCK:
        return s.evaluate(n=max(1, min(300, int(req.n))), k=req.k)


class PlayReq(BaseModel):
    k: int | None = None


@app.post("/api/training/play")
def post_play(req: PlayReq):
    """Scramble + solve one cube and return every frame (cube state) plus the
    per-step move and heuristic scores — the 3D Watch tab animates this."""
    s = _require_session()
    with SESSION_LOCK:
        return s.solve_episode(k=req.k)


# ── Checkpoints ────────────────────────────────────────────────────────
class CheckpointReq(BaseModel):
    filename: str


@app.get("/api/training/checkpoints")
def get_checkpoints():
    return {"files": _training.list_checkpoints()}


@app.post("/api/training/checkpoints/save")
def post_save_checkpoint(req: CheckpointReq):
    s = _require_session()
    with SESSION_LOCK:
        return {"name": _training.save_checkpoint(s, req.filename)}


@app.post("/api/training/checkpoints/delete")
def post_delete_checkpoint(req: CheckpointReq):
    try:
        return {"name": _training.delete_checkpoint(req.filename)}
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/training/checkpoints/load")
async def post_load_checkpoint(req: CheckpointReq):
    _refuse_if_running()
    try:
        with SESSION_LOCK:
            session = _training.load_checkpoint(req.filename, device)
            _training_state["session"] = session
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    from dataclasses import asdict
    new_state = _agent_state.get_state()
    new_state["algorithm"] = {"algo": session.algo, "hyperparameters": asdict(session.hp)}
    new_state["environment"]["size"] = session.size
    await _agent_state.replace_state(new_state, source="ui")
    return {**_session_summary(session), "loss_history": session.loss_history}


# ── Background training run ─────────────────────────────────────────────
class RunStartReq(BaseModel):
    start_k: int | None = None
    max_k: int | None = None
    promote_at: float | None = None
    max_iterations: int | None = None
    eval_every: int | None = None
    checkpoint_every: int | None = None
    cadence_minutes: float | None = None


def _run_config_from_state(req: RunStartReq) -> RunConfig:
    state = _agent_state.get_state()
    env = state["environment"]
    cur = env.get("curriculum", {})
    tr = state["training"]
    return RunConfig(
        cube_size=int(env["size"]),
        start_k=int(req.start_k if req.start_k is not None else cur.get("startK", 1)),
        max_k=int(req.max_k if req.max_k is not None else cur.get("maxK", 14)),
        promote_at=float(req.promote_at if req.promote_at is not None else cur.get("promoteAt", 0.9)),
        max_iterations=int(req.max_iterations or 0),
        eval_every=int(req.eval_every if req.eval_every is not None else tr.get("evalEveryN", 100)),
        eval_n=int(tr.get("evalN", 80)),
        checkpoint_every=int(req.checkpoint_every if req.checkpoint_every is not None else 200),
        cadence_minutes=float(req.cadence_minutes if req.cadence_minutes is not None
                              else tr.get("cadenceMinutes", 20)),
    )


@app.post("/api/training/run/start")
def post_run_start(req: RunStartReq):
    try:
        status = trainer.start(_run_config_from_state(req), fresh=True)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e))
    return status


@app.post("/api/training/run/stop")
def post_run_stop():
    return trainer.stop()


@app.get("/api/training/run/status")
def get_run_status():
    return trainer.status()


@app.post("/api/training/run/resume")
def post_run_resume():
    m = _bg.find_resumable()
    if m is None:
        raise HTTPException(404, "no resumable run found")
    try:
        with SESSION_LOCK:
            session = _training.load_checkpoint(m["checkpoint"], device)
            _training_state["session"] = session
        cfg = RunConfig(**{k: v for k, v in m["config"].items()
                           if k in RunConfig.__dataclass_fields__})
        status = trainer.start(cfg, fresh=False, run_id=m["run_id"])
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise HTTPException(409, str(e))
    return status


@app.get("/api/training/report")
def get_report():
    return _report_store.get()


# ── Pipeline-state mirror ──────────────────────────────────────────────
class StatePatchReq(BaseModel):
    patch: dict


@app.get("/api/state")
def get_pipeline_state():
    return _agent_state.get_state()


@app.post("/api/state/patch")
async def post_state_patch(req: StatePatchReq):
    await _agent_state.apply_patch(req.patch, source="ui")
    return {"ok": True}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    queue, unsubscribe = _agent_state.subscribe()
    try:
        await websocket.send_json({
            "type": "state_replace", "source": "system",
            "state": _agent_state.get_state(),
        })
        while True:
            ev = await queue.get()
            await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()


# ── Startup: capture the loop, launch the scheduler ────────────────────
@app.on_event("startup")
async def _on_startup():
    _agent_state.set_loop(asyncio.get_running_loop())
    scheduler.start()


# ── Agent chat ─────────────────────────────────────────────────────────
class AgentChatReq(BaseModel):
    message: str
    session_id: str | None = None


_current_agent_task: dict[str, asyncio.Task | None] = {"task": None}


@app.post("/api/agent/chat")
async def post_agent_chat(req: AgentChatReq):
    prev = _current_agent_task["task"]
    if prev is not None and not prev.done():
        prev.cancel()

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()

        async def producer():
            # Serialize against autonomous check-ins on the shared coach session.
            async with _agent_runtime.AGENT_TURN_LOCK:
                try:
                    async for ev in _agent_runtime.run_turn(
                        req.message, resume_session_id=req.session_id,
                    ):
                        if ev.get("type") == "result" and ev.get("session_id"):
                            _coach_session["id"] = ev["session_id"]
                        await queue.put(ev)
                finally:
                    await queue.put(DONE)

        task = asyncio.create_task(producer())
        _current_agent_task["task"] = task
        try:
            while True:
                ev = await queue.get()
                if ev is DONE:
                    break
                yield f"data: {json.dumps(ev)}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            yield f"data: {json.dumps({'type': 'error', 'message': 'cancelled'})}\n\n"
            raise
        finally:
            if not task.done():
                task.cancel()
            if _current_agent_task["task"] is task:
                _current_agent_task["task"] = None

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/agent/stop")
def post_agent_stop():
    task = _current_agent_task["task"]
    if task is None or task.done():
        return {"cancelled": False, "reason": "no in-flight turn"}
    task.cancel()
    return {"cancelled": True}


@app.get("/api/agent/sessions")
def get_agent_sessions(limit: int = 50):
    from claude_agent_sdk import list_sessions
    sessions = list_sessions(directory=str(_HERE), limit=limit)
    out = []
    for s in sessions:
        out.append({
            "session_id": getattr(s, "session_id", None),
            "summary": getattr(s, "summary", None),
            "message_count": getattr(s, "message_count", None),
            "last_modified": getattr(s, "last_modified", None),
            "created_at": getattr(s, "created_at", None),
        })
    return {"sessions": out}


@app.get("/api/agent/sessions/{session_id}/messages")
def get_agent_session_messages(session_id: str):
    from claude_agent_sdk import get_session_messages
    try:
        raw = get_session_messages(session_id=session_id, directory=str(_HERE))
    except Exception as e:
        raise HTTPException(404, f"session not found: {e}")

    events: list[dict] = []
    for msg in raw:
        m = msg if isinstance(msg, dict) else getattr(msg, "__dict__", {}) or {}
        msg_type = m.get("type")
        body = m.get("message") or {}
        content = body.get("content") if isinstance(body, dict) else None
        if msg_type == "user":
            if isinstance(content, str):
                events.append({"type": "user_message", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            events.append({"type": "user_message",
                                           "text": block.get("text", "")})
                        elif block.get("type") == "tool_result":
                            events.append({
                                "type": "tool_result",
                                "tool_use_id": block.get("tool_use_id"),
                                "content": _flatten_tool_result_content(
                                    block.get("content")),
                                "is_error": bool(block.get("is_error", False)),
                            })
        elif msg_type == "assistant":
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")
                    if bt == "text":
                        events.append({"type": "text_message",
                                       "text": block.get("text", "")})
                    elif bt == "tool_use":
                        full = block.get("name", "")
                        short = full.split("__")[-1] if full else full
                        events.append({
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": short,
                            "full_name": full,
                            "input": block.get("input"),
                        })
    return {"session_id": session_id, "events": events}


def _flatten_tool_result_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, str):
                parts.append(b)
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(content)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    os.chdir(_HERE)
    port = int(os.environ.get("CUBE_SERVER_PORT", "5041"))
    # Overnight runs must disable reload: a file save would kill the background
    # training thread. Set CUBE_NO_RELOAD=1 for long/unattended runs.
    reload = os.environ.get("CUBE_NO_RELOAD", "") not in ("1", "true", "yes")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=reload,
                reload_dirs=[str(_HERE)], log_level="info")

"""FastAPI backend for the Agentic Snake Trainer.

Same architecture as 04-agentic-symbols: a shared pipeline-state mirror that
both the UI and the RL Coach agent drive, a /ws/state socket that broadcasts
state patches + live training ticks, and an SSE agent-chat endpoint. The
domain underneath is reinforcement learning (a SnakeSession of env + agent)
rather than supervised symbol OCR.

Endpoints
  GET  /api/device  ·  GET /api/device/list  ·  POST /api/device/select
  GET  /api/catalog                         algorithms + their default hyperparameters

  GET  /api/state  ·  POST /api/state/patch  ·  WS /ws/state

  POST /api/training/init                   build a session from the state mirror
  GET  /api/training/state
  POST /api/training/reset
  POST /api/training/train_episodes         run a bounded chunk of episodes
  POST /api/training/eval                   greedy evaluation (no learning)
  POST /api/training/play                   one game, with frames + per-action scores
  GET  /api/training/checkpoints  ·  save  ·  load  ·  delete

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
import training as _training  # noqa: E402
from agents import ALGORITHMS, default_hyperparams  # noqa: E402
from models import pick_device  # noqa: E402

STATIC_DIR = _HERE / "frontend" / "dist"

device = pick_device()

app = FastAPI(title="Agentic Snake Trainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Device info (ported from 04; macOS-aware) ──────────────────────────
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
    """Swap the global device and migrate a live session's tensors. Shared
    by the REST endpoint and the agent's select_device tool."""
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
    """Algorithm metadata + per-algorithm default hyperparameters, so the
    Algorithm tab can render the picker and seed the sliders."""
    algos = []
    for key, meta in ALGORITHMS.items():
        algos.append({"id": key, **meta, "default_hyperparameters": default_hyperparams(key)})
    return {"algorithms": algos}


# ── Training session ───────────────────────────────────────────────────
_training_state: dict = {"session": None}

_agent_tools.set_session_accessors(
    get_session=lambda: _training_state["session"],
    set_session=lambda s: _training_state.__setitem__("session", s),
    get_device=lambda: device,
    set_device=_select_device_impl,
)


def _require_session() -> _training.SnakeSession:
    s = _training_state["session"]
    if s is None:
        raise HTTPException(409, "no training session — initialize first")
    return s


def _session_summary(s: _training.SnakeSession) -> dict:
    return {
        "has_session": True,
        "algo": s.algo,
        "uses_network": s.uses_network(),
        "param_count": s.param_count(),
        "episode": s.episode,
        "best_score": s.best_score,
        "device": str(s.device),
    }


@app.post("/api/training/init")
def post_training_init():
    """Build a fresh session from the current pipeline-state mirror
    (algorithm + hyperparameters + environment). Replaces any existing
    session. Mirrors the agent's init_session tool — same source of truth."""
    state = _agent_state.get_state()
    algo = state["algorithm"]["algo"]
    if algo not in ALGORITHMS:
        raise HTTPException(400, f"unknown algorithm: {algo}")
    try:
        session = _training.SnakeSession(
            algo,
            state["algorithm"]["hyperparameters"],
            state["environment"],
            device,
        )
    except ValueError as e:
        # e.g. tabular Q-learning requested on the grid observation.
        raise HTTPException(400, str(e))
    _training_state["session"] = session
    # No training_session broadcast here: the initiating tab updates itself
    # from this response, and broadcasting an empty-history reset back could
    # race its first pushed episodes. Only the agent path broadcasts (so the
    # UI reflects agent-driven init/reset/load it didn't initiate).
    return _session_summary(session)


@app.get("/api/training/state")
def get_training_state():
    s = _training_state["session"]
    if s is None:
        return {"has_session": False}
    return {**_session_summary(s), "score_history": s.score_history}


@app.post("/api/training/reset")
def post_training_reset():
    _training_state["session"] = None
    return {"has_session": False}


class TrainEpisodesReq(BaseModel):
    n: int = 5
    # Optional hot-swap hyperparameters applied before the chunk runs.
    hyperparameters: dict | None = None


@app.post("/api/training/train_episodes")
def post_train_episodes(req: TrainEpisodesReq):
    """Run a bounded chunk of episodes and return one record per episode.
    The frontend calls this in a loop (with a Stop flag) to train for many
    episodes while keeping the score chart live."""
    s = _require_session()
    if req.hyperparameters:
        s.update_hyperparameters(req.hyperparameters)
    n = max(1, min(100, int(req.n)))
    records: list[dict] = []
    summary = s.train_episodes(n, on_episode=records.append)
    return {"records": records, "summary": summary}


class EvalReq(BaseModel):
    n: int = 20


@app.post("/api/training/eval")
def post_eval(req: EvalReq):
    s = _require_session()
    return s.evaluate(max(1, min(200, int(req.n))))


class PlayReq(BaseModel):
    greedy: bool = True


@app.post("/api/training/play")
def post_play(req: PlayReq):
    """Run one episode and return every frame plus the agent's per-action
    scores — the Watch tab animates this."""
    s = _require_session()
    return s.play_episode(greedy=req.greedy)


# ── Checkpoints ────────────────────────────────────────────────────────
class CheckpointReq(BaseModel):
    filename: str


@app.get("/api/training/checkpoints")
def get_checkpoints():
    return {"files": _training.list_checkpoints()}


@app.post("/api/training/checkpoints/save")
def post_save_checkpoint(req: CheckpointReq):
    s = _require_session()
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
    try:
        session = _training.load_checkpoint(req.filename, device)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    _training_state["session"] = session
    # Restore the pipeline mirror so the UI reflects what the model trained
    # against, then push a session-restore event with the score history.
    new_state = _agent_state.get_state()
    new_state["algorithm"] = {
        "algo": session.algo,
        "hyperparameters": {k: v for k, v in _dataclass_dict(session.hp).items()},
    }
    new_state["environment"] = _training.env_config_to_dict(session.env_cfg)
    await _agent_state.replace_state(new_state, source="ui")
    # As with init/reset: the initiating tab applies this response directly;
    # only env/algo sliders need the state_replace broadcast above.
    return {**_session_summary(session), "score_history": session.score_history}


def _dataclass_dict(hp) -> dict:
    from dataclasses import asdict
    return asdict(hp)


# ── Pipeline-state mirror (ported from 04) ─────────────────────────────
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


# ── Agent chat (ported from 04) ────────────────────────────────────────
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
            try:
                async for ev in _agent_runtime.run_turn(
                    req.message, resume_session_id=req.session_id,
                ):
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
    port = int(os.environ.get("SNAKE_SERVER_PORT", "5041"))
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True,
                reload_dirs=[str(_HERE)], log_level="info")

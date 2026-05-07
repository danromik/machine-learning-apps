"""FastAPI backend for the Agentic Symbol Trainer.

Endpoints (existing):
  GET  /api/device
  GET  /api/synthesis/symbols
  GET  /api/synthesis/fonts
  GET  /api/synthesis/preset/{name}     name in {beginner, intermediate, advanced}
  POST /api/synthesis/sample            streaming NDJSON of synthesized samples

  POST /api/training/init               build a fresh training session
  GET  /api/training/state              current session status
  POST /api/training/predict            forward-only on a list of base64 PNGs
  POST /api/training/train_batch        one forward + backward + step
  GET  /api/training/checkpoints         list checkpoint files in checkpoints/
  POST /api/training/checkpoints/save    save current session under a filename
  POST /api/training/checkpoints/load    load a session from a checkpoint file
  POST /api/training/checkpoints/delete  remove a checkpoint file from disk

Endpoints (agent / state):
  GET  /api/state                            current pipeline-state mirror
  POST /api/state/patch                      apply a patch from the UI
  WS   /ws/state                             stream of pipeline-state patches

  POST /api/agent/chat                       run one ML Engineer turn (SSE stream)
  POST /api/agent/stop                       interrupt current turn
  GET  /api/agent/sessions                   list past chat sessions
  GET  /api/agent/sessions/{id}/messages     full transcript of a past session
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

import ml as _ml  # noqa: E402
import synthesis as _synthesis  # noqa: E402
import training as _training  # noqa: E402
import agent_state as _agent_state  # noqa: E402
import agent_runtime as _agent_runtime  # noqa: E402
import agent_tools as _agent_tools  # noqa: E402

STATIC_DIR = _HERE / "frontend" / "dist"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = pick_device()

app = FastAPI(title="Agentic Symbol Trainer")

# Vite dev server runs on a different port, so allow CORS in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sysctl(name: str) -> str | None:
    """Best-effort macOS sysctl lookup; None on any failure."""
    import subprocess

    try:
        r = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2
        )
        out = r.stdout.strip()
        return out or None
    except Exception:
        return None


def _system_memory_bytes() -> int | None:
    """Total physical memory, in bytes. macOS via sysctl; Linux via /proc."""
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
    """Return display info for a torch device key ('cpu' / 'mps' / 'cuda').

    Memory is unified on Apple Silicon, so cpu and mps both report the
    system's total RAM; the mps entry includes a `memory_note` so the UI
    can flag that.
    """
    info: dict = {"name": name}
    mem = _system_memory_bytes()
    if name == "cpu":
        info["label"] = (
            _sysctl("machdep.cpu.brand_string")
            or platform.processor()
            or platform.machine()
            or "CPU"
        )
        cores = _sysctl("hw.ncpu")
        if cores:
            try:
                info["cores"] = int(cores)
            except ValueError:
                pass
        clock = _sysctl("hw.cpufrequency_max")  # only set on older Intel macs
        if clock:
            try:
                info["clock_hz"] = int(clock)
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
            # CUDA exposes a SM clock rate in kHz via properties on newer
            # PyTorch versions; fall back if unavailable.
            clock_khz = getattr(props, "clock_rate", None)
            if clock_khz:
                info["clock_hz"] = int(clock_khz) * 1000
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
    name: str  # 'cpu' | 'mps' | 'cuda'


def _select_device_impl(name: str) -> torch.device:
    """Validate the device name, swap the global, and migrate the live
    session's tensors. Returns the new device. Used by the REST endpoint
    AND by the agent's `select_device` tool — same code path so behavior
    can't drift."""
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
        # Move any live session to the new device. Optimizer state has
        # tensors of its own (Adam moments, etc.) that aren't covered by
        # model.to() — walk and move them by hand.
        s = _training_state["session"]
        if s is not None:
            s.model.to(device)
            for st in s.optimizer.state.values():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(device)
            s.device = device
    return device


@app.post("/api/device/select")
def post_device_select(req: DeviceSelectReq):
    try:
        new_device = _select_device_impl(req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"current": str(new_device)}


@app.get("/api/synthesis/symbols")
def get_symbols():
    return {"categories": _ml.list_categories()}


@app.get("/api/synthesis/fonts")
def get_fonts():
    return {"fonts": _ml.list_curated_fonts()}


@app.get("/api/synthesis/preset/{name}")
def get_preset(name: str):
    try:
        return _ml.get_preset(name)
    except ValueError as e:
        raise HTTPException(404, str(e))


class AugmentationCfg(BaseModel):
    noise: dict
    skew: dict


class SampleReq(BaseModel):
    categories: list[str]
    training_fonts: list[str]
    validation_fonts: list[str]
    augmentation: AugmentationCfg
    split: str  # 'train' or 'val'
    count: int = 500
    seed: int = 42


class InferenceReq(BaseModel):
    chars: list[str]
    # Optional preferred font; falls back through `fonts` and finally to
    # any installed curated font that supports the glyph.
    font: str | None = None
    fonts: list[str] | None = None


@app.post("/api/inference/render")
def post_inference_render(req: InferenceReq):
    """Render each input char to a 64×64 grayscale PNG through the
    training pipeline and (if a training session is loaded) classify it,
    re-rendering the predicted class so the frontend can show side-by-
    side glyph grids of input vs prediction without extra round-trips.
    """
    import base64
    import io

    candidates: list[str] = []
    if req.font:
        candidates.append(req.font)
    if req.fonts:
        candidates.extend(f for f in req.fonts if f != req.font)
    # Final fallback: any installed curated font, in priority order.
    candidates.extend(
        f["family"]
        for f in _ml.list_curated_fonts()
        if f["installed"] and f["family"] not in candidates
    )

    def render_one(ch: str) -> tuple[str | None, str | None]:
        if not ch:
            return None, None
        for family in candidates:
            path = _ml.font_path(family)
            if path is None or not _synthesis._has_glyph(ch, path):
                continue
            img = _synthesis.render_glyph(ch, path)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("ascii"), family
        return None, None

    # Render inputs.
    input_renders = [render_one(ch) for ch in req.chars]

    # Batch-predict the renders that exist (skip empty/unsupported chars).
    s = _training_state["session"]
    pred_by_idx: dict[int, list[float]] = {}
    if s is not None:
        valid_idx = [i for i, (png, _) in enumerate(input_renders) if png]
        if valid_idx:
            pngs = [input_renders[i][0] for i in valid_idx]  # type: ignore
            all_probs = s.predict(pngs)
            for k, i in enumerate(valid_idx):
                pred_by_idx[i] = all_probs[k]

    # K matches the on-screen capacity of the per-glyph detail panel —
    # 5 alternatives is enough to show the confusables (top-1 + a few
    # near-tie classes) without flooding the UI.
    TOP_K = 5

    items: list[dict] = []
    for i, ch in enumerate(req.chars):
        png, font = input_renders[i]
        item: dict = {
            "char": ch,
            "input_png_b64": png,
            "input_font": font,
            "predicted_char": None,
            "predicted_png_b64": None,
            "predicted_font": None,
            "confidence": None,
            "in_class_set": (s is not None) and (ch in s.label_to_index)
            if s is not None
            else False,
            "top_k": [],
        }
        if i in pred_by_idx:
            probs = pred_by_idx[i]
            assert s is not None
            # Get top-K (idx, prob) pairs, descending by prob.
            ranked = sorted(
                enumerate(probs), key=lambda x: -x[1]
            )[:TOP_K]
            top_k = []
            for idx, prob in ranked:
                cls = s.classes[idx]
                cls_png, cls_font = render_one(cls)
                top_k.append(
                    {
                        "char": cls,
                        "png_b64": cls_png,
                        "font": cls_font,
                        "confidence": float(prob),
                    }
                )
            item["top_k"] = top_k
            # Top-1 convenience fields, kept for the existing grid UI.
            item["predicted_char"] = top_k[0]["char"]
            item["predicted_png_b64"] = top_k[0]["png_b64"]
            item["predicted_font"] = top_k[0]["font"]
            item["confidence"] = top_k[0]["confidence"]
        items.append(item)
    return {"items": items, "has_session": s is not None}


@app.post("/api/synthesis/sample")
def post_sample(req: SampleReq):
    if req.split not in ("train", "val"):
        raise HTTPException(400, "split must be 'train' or 'val'")
    cfg = _synthesis.SynthesisRequest(
        categories=req.categories,
        training_fonts=req.training_fonts,
        validation_fonts=req.validation_fonts,
        augmentation={
            "noise": req.augmentation.noise,
            "skew": req.augmentation.skew,
        },
        split=req.split,
        count=req.count,
        seed=req.seed,
    )

    def stream():
        for sample in _synthesis.synthesize_iter(cfg):
            yield json.dumps(sample) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class TrainingHyperparameters(BaseModel):
    lr: float
    batch_size: int
    optimizer: str


class InitTrainingReq(BaseModel):
    architecture: list[dict]
    hyperparameters: TrainingHyperparameters
    classes: list[str]
    # Snapshot of the synthesis config (selectedCategories, fontUsage,
    # augmentation) the model is being built against. Optional because
    # older callers may not send it; checkpoint save/load rely on it
    # being present.
    synthesis_config: dict | None = None


class PredictReq(BaseModel):
    images: list[str]


class TrainBatchReq(BaseModel):
    images: list[str]
    labels: list[str]
    # Optional hot-swap hyperparameters. When present, the session updates
    # its optimizer (rebuilds on optimizer name change, in-place lr swap
    # otherwise) before the gradient step.
    lr: float | None = None
    optimizer: str | None = None


class CheckpointReq(BaseModel):
    filename: str


# Single global training session. Mirrors the MNIST trainer's pattern:
# one model lives at a time; re-initializing replaces it.
_training_state: dict = {"session": None}


# Wire the agent tools' session/device accessors. Done here (rather than
# at agent_tools import time) because the helpers close over module-level
# globals that need to exist first.
_agent_tools.set_session_accessors(
    get_session=lambda: _training_state["session"],
    set_session=lambda s: _training_state.__setitem__("session", s),
    get_device=lambda: device,
    set_device=_select_device_impl,
)


def _require_session() -> _training.TrainingSession:
    s = _training_state["session"]
    if s is None:
        raise HTTPException(409, "no training session — initialize first")
    return s


def _session_summary(s: _training.TrainingSession) -> dict:
    return {
        "has_session": True,
        "num_classes": s.num_classes,
        "param_count": s.param_count(),
        "step": s.step,
        "lr": s.lr,
        "batch_size": s.batch_size,
        "optimizer": s.optimizer_name,
    }


@app.post("/api/training/init")
def post_training_init(req: InitTrainingReq):
    try:
        session = _training.TrainingSession(
            req.architecture,
            req.hyperparameters.model_dump(),
            req.classes,
            device,
            synthesis_config=req.synthesis_config,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    _training_state["session"] = session
    return _session_summary(session)


@app.get("/api/training/state")
def get_training_state():
    s = _training_state["session"]
    if s is None:
        return {"has_session": False}
    return _session_summary(s)


@app.post("/api/training/reset")
def post_training_reset():
    """Drop the live session. Called by the frontend whenever the data
    synthesis config changes — the model's class set is tied to that
    config, so the previous session would be invalid against any new
    batch."""
    _training_state["session"] = None
    return {"has_session": False}


@app.post("/api/training/predict")
def post_predict(req: PredictReq):
    s = _require_session()
    return {"predictions": s.predict(req.images)}


@app.post("/api/training/eval")
def post_eval(req: TrainBatchReq):
    """Forward-only loss + accuracy on a held-out batch. Same payload
    shape as train_batch (minus optional hyperparam fields, which are
    ignored here) so the frontend can reuse the encoding path."""
    s = _require_session()
    try:
        result = s.eval_batch(req.images, req.labels)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.post("/api/training/train_batch")
def post_train_batch(req: TrainBatchReq):
    s = _require_session()
    try:
        result = s.train_batch(
            req.images, req.labels, lr=req.lr, optimizer_name=req.optimizer
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/api/training/checkpoints")
def get_checkpoints():
    return {"files": _training.list_checkpoints()}


@app.post("/api/training/checkpoints/save")
def post_save_checkpoint(req: CheckpointReq):
    s = _require_session()
    name = _training.save_checkpoint(s, req.filename)
    return {"name": name}


@app.post("/api/training/checkpoints/delete")
def post_delete_checkpoint(req: CheckpointReq):
    try:
        name = _training.delete_checkpoint(req.filename)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"name": name}


@app.post("/api/training/checkpoints/load")
def post_load_checkpoint(req: CheckpointReq):
    try:
        session = _training.load_checkpoint(req.filename, device)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    _training_state["session"] = session
    # Return the full pipeline state so the frontend can restore the
    # synthesis tab, architecture diagram, hyperparameter sliders, and
    # the training/validation loss curves to match the loaded model.
    return {
        **_session_summary(session),
        "layers": session.layers,
        "classes": session.classes,
        "synthesis_config": session.synthesis_config,
        "loss_history": session.loss_history,
        "val_loss_history": session.val_loss_history,
    }


# ── Pipeline state mirror (shared between UI and agent) ───────────────


class StatePatchReq(BaseModel):
    patch: dict


@app.get("/api/state")
def get_pipeline_state():
    return _agent_state.get_state()


@app.post("/api/state/patch")
async def post_state_patch(req: StatePatchReq):
    """Frontend pushes a patch (e.g. user changed a slider). The patch is
    deep-merged into the backend mirror and broadcast to all WS
    subscribers — including this caller's other tabs, but tagged
    `source: 'ui'` so the originating tab can ignore the echo."""
    await _agent_state.apply_patch(req.patch, source="ui")
    return {"ok": True}


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    """Push pipeline-state patches to the client as they happen.
    Initial message is a full state snapshot so the client doesn't have
    to race a separate /api/state fetch against the first patch."""
    await websocket.accept()
    queue, unsubscribe = _agent_state.subscribe()
    try:
        await websocket.send_json({
            "type": "state_replace",
            "source": "system",
            "state": _agent_state.get_state(),
        })
        while True:
            ev = await queue.get()
            await websocket.send_json(ev)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()


# ── Agent chat ────────────────────────────────────────────────────────


class AgentChatReq(BaseModel):
    message: str
    session_id: str | None = None  # resume an existing session if provided


# Reference to the currently-running agent turn task so /api/agent/stop
# can cancel it. Single-user app, single in-flight turn.
_current_agent_task: dict[str, asyncio.Task | None] = {"task": None}


@app.post("/api/agent/chat")
async def post_agent_chat(req: AgentChatReq):
    """Run one ML Engineer turn. Returns an SSE stream of normalized
    events (text deltas, tool calls, tool results, usage, final result).
    The frontend reduces these into chat bubbles + tool info rows.

    Concurrency model: only one turn at a time. If a turn is already in
    flight, this endpoint replaces it (the old one is cancelled) — same
    pattern as the user clicking Send twice.
    """
    # Cancel any in-flight turn before starting a new one.
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
            # Client disconnected mid-stream. Cancel the producer too.
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
    """Cancel the in-flight agent turn, if any. Returns whether anything
    was actually cancelled."""
    task = _current_agent_task["task"]
    if task is None or task.done():
        return {"cancelled": False, "reason": "no in-flight turn"}
    task.cancel()
    return {"cancelled": True}


@app.get("/api/agent/sessions")
def get_agent_sessions(limit: int = 50):
    """List past chat sessions (most recent first). Each entry has
    session_id, summary, message_count, last_modified."""
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
    """Full message history for a past session — used to reconstruct the
    chat transcript when the user resumes a previous conversation. Each
    raw SDK message is normalized into our event shape so the frontend
    renderer can be fed historical events with the same code path it
    uses for live streams."""
    from claude_agent_sdk import get_session_messages
    try:
        raw = get_session_messages(session_id=session_id, directory=str(_HERE))
    except Exception as e:
        raise HTTPException(404, f"session not found: {e}")

    events: list[dict] = []
    for msg in raw:
        m = msg if isinstance(msg, dict) else getattr(msg, "__dict__", {}) or {}
        # SessionMessage shape: {type: "user"|"assistant", message: {...}}
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


# Serve built frontend if present.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    port = int(os.environ.get("AGENTIC_SERVER_PORT", "5041"))
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(here)],
        log_level="info",
    )

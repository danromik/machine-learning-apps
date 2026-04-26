"""FastAPI backend for the MNIST web app.

Endpoints:
  GET  /api/device
  GET  /api/params?model=cnn|mlp
  GET  /api/counts?class_filter=<int|null>
  GET  /api/sample?split=train|test&class=all|<int>&order=default|digit&n=200
  GET  /api/checkpoints
  POST /api/checkpoints/load       {name}
  POST /api/checkpoints/save
  POST /api/train/start            {model, epochs, batch_size, lr, seed, max_steps?, max_epochs?}
  POST /api/train/stop
  POST /api/session/reset
  POST /api/autosave               {enabled}
  POST /api/predict                {data_url}
  WS   /ws                         streams training events
"""
from __future__ import annotations

import asyncio
import queue
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Sibling-import ml/training_worker. Make sure this file's directory is on
# sys.path so `import ml` works when uvicorn imports `server:app`.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ml as _ml  # noqa: E402
import training_worker as _tw  # noqa: E402

STATIC_DIR = _HERE / "frontend" / "dist"

TrainConfig = _tw.TrainConfig
TrainingWorker = _tw.TrainingWorker

worker = TrainingWorker()
device = _ml.pick_device()
_state: dict = {"model": None, "ckpt_name": None}

app = FastAPI(title="MNIST Trainer")

# Vite dev server runs on a different port, so allow CORS in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────
class StartReq(BaseModel):
    model: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None


class PredictReq(BaseModel):
    data_url: str


class TrainCfgBase(BaseModel):
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0


class ResetReq(BaseModel):
    model: str = "cnn"
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0


class LoadReq(BaseModel):
    name: str
    cfg: Optional[TrainCfgBase] = None


class AutoSaveReq(BaseModel):
    enabled: bool


# ── REST ────────────────────────────────────────────────────────────────
@app.get("/api/device")
def get_device():
    return {"device": str(device)}


@app.get("/api/params")
def get_params(model: str):
    return {"params": _ml.count_params(model)}


@app.get("/api/architecture")
def get_architecture(model: str):
    try:
        return _ml.describe_architecture(model)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/counts")
def get_counts(class_filter: Optional[int] = None):
    return _ml.count_mnist(class_filter)


@app.get("/api/sample")
def get_sample(
    split: str = "train",
    class_: str = "all",
    order: str = "default",
    n: int = 200,
    offset: int = 0,
):
    cls = None if class_ == "all" else int(class_)
    samples = _ml.sample_mnist(split, cls, n, order, offset=offset)
    total = _ml.count_mnist(cls)["train" if split == "train" else "test"]
    return {"samples": samples, "offset": offset, "total": total}


# Query-params can't use 'class' as a name; accept it via alias.
@app.get("/api/sample_q")
def get_sample_q(
    split: str = "train",
    cls: Optional[str] = "all",
    order: str = "default",
    n: int = 200,
    offset: int = 0,
):
    c = None if cls in (None, "all") else int(cls)
    samples = _ml.sample_mnist(split, c, n, order, offset=offset)
    total = _ml.count_mnist(c)["train" if split == "train" else "test"]
    return {"samples": samples, "offset": offset, "total": total}


@app.get("/api/checkpoints")
def get_checkpoints():
    return {"files": _ml.list_checkpoints(), "current": _state["ckpt_name"]}


@app.post("/api/checkpoints/load")
def post_load(req: LoadReq):
    import torch as _torch

    ckpt_path = _ml.CKPT_DIR / req.name
    if not ckpt_path.exists():
        raise HTTPException(404, f"no such checkpoint: {req.name}")
    if worker.running:
        raise HTTPException(409, "stop training before loading a checkpoint")
    ckpt = _torch.load(ckpt_path, map_location=device, weights_only=False)
    base = req.cfg or TrainCfgBase()
    cfg = TrainConfig(
        model=ckpt.get("model", "cnn"),
        epochs=base.epochs,
        batch_size=base.batch_size,
        lr=base.lr,
        seed=base.seed,
    )
    info = worker.load_into_session(ckpt, cfg)
    _state["ckpt_name"] = req.name
    _state["model"] = worker._session["model"]
    return {"name": req.name, **info}


@app.get("/api/session")
def get_session_state():
    return worker.session_state()


@app.post("/api/checkpoints/save")
def post_save():
    name = worker.save_checkpoint()
    if not name:
        raise HTTPException(409, "no session — train at least one step first")
    _state["ckpt_name"] = name
    return {"name": name}


@app.post("/api/train/start")
def post_start(req: StartReq):
    cfg = TrainConfig(
        model=req.model,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        seed=req.seed,
    )
    worker.start_run(cfg, max_steps=req.max_steps, max_epochs=req.max_epochs)
    return {"ok": True}


@app.post("/api/train/stop")
def post_stop():
    worker.stop()
    return {"ok": True}


@app.post("/api/session/reset")
def post_reset(req: ResetReq = ResetReq()):
    if worker.running:
        raise HTTPException(409, "stop training before re-initializing")
    cfg = TrainConfig(
        model=req.model,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        seed=req.seed,
    )
    worker.reset_session(cfg)
    # Stale cached model from a prior load/predict must not shadow the fresh
    # random-weights session for inference.
    _state["model"] = None
    _state["ckpt_name"] = None
    return {"ok": True}


@app.post("/api/autosave")
def post_autosave(req: AutoSaveReq):
    worker.auto_save_checkpoint = bool(req.enabled)
    return {"enabled": worker.auto_save_checkpoint}


@app.post("/api/predict")
def post_predict(req: PredictReq):
    # Prefer the live training model if there's an active session.
    if worker._session is not None:
        model = worker._session["model"]
    elif _state["model"] is not None:
        model = _state["model"]
    else:
        files = _ml.list_checkpoints()
        if not files:
            raise HTTPException(409, "no model loaded and no checkpoints available")
        _state["model"] = _ml.load_checkpoint(files[0], device)
        _state["ckpt_name"] = files[0]
        model = _state["model"]
    result = _ml.predict_from_data_url(req.data_url, model, device)
    preview_b64 = _ml.render_canvas_preview(req.data_url)
    return {
        "pred": result["pred"],
        "probs": result["probs"],
        "preview_b64": preview_b64,
        "ckpt_name": _state["ckpt_name"],
    }


# ── WebSocket ────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                ev = await asyncio.to_thread(worker.events.get, True, 0.2)
            except queue.Empty:
                # Use this tick to verify the client is still connected.
                try:
                    await asyncio.wait_for(ws.send_json({"type": "ping"}), timeout=1.0)
                except Exception:
                    return
                continue
            await ws.send_json(ev)
    except WebSocketDisconnect:
        return


# ── Serve built frontend (if present) ────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    # Run `uv run python 01-mnist-trainer/server.py` from anywhere — we cd
    # into this file's directory so uvicorn's reloader can re-import "server".
    import os
    import uvicorn

    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    port = int(os.environ.get("MNIST_SERVER_PORT", "5041"))
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(here)],
        log_level="info",
    )

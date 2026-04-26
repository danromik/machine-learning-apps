"""FastAPI backend for the Math Symbol Trainer.

Endpoints:
  GET  /api/device
  GET  /api/synthesis/symbols
  GET  /api/synthesis/fonts
  GET  /api/synthesis/preset/{name}     name in {beginner, intermediate, advanced}
  POST /api/synthesis/sample            streaming NDJSON of synthesized samples

  POST /api/training/init               build a fresh training session
  GET  /api/training/state              current session status
  POST /api/training/predict            forward-only on a list of base64 PNGs
  POST /api/training/train_batch        one forward + backward + step
  GET  /api/training/checkpoints        list checkpoint files in checkpoints/
  POST /api/training/checkpoints/save   save current session under a filename
  POST /api/training/checkpoints/load   load a session from a checkpoint file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
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

STATIC_DIR = _HERE / "frontend" / "dist"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = pick_device()

app = FastAPI(title="Math Symbol Trainer")

# Vite dev server runs on a different port, so allow CORS in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/device")
def get_device():
    return {"device": str(device)}


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


class PredictReq(BaseModel):
    images: list[str]


class TrainBatchReq(BaseModel):
    images: list[str]
    labels: list[str]


class CheckpointReq(BaseModel):
    filename: str


# Single global training session. Mirrors the MNIST trainer's pattern:
# one model lives at a time; re-initializing replaces it.
_training_state: dict = {"session": None}


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


@app.post("/api/training/predict")
def post_predict(req: PredictReq):
    s = _require_session()
    return {"predictions": s.predict(req.images)}


@app.post("/api/training/train_batch")
def post_train_batch(req: TrainBatchReq):
    s = _require_session()
    try:
        result = s.train_batch(req.images, req.labels)
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


@app.post("/api/training/checkpoints/load")
def post_load_checkpoint(req: CheckpointReq):
    try:
        session = _training.load_checkpoint(req.filename, device)
    except FileNotFoundError as e:
        raise HTTPException(404, f"no such checkpoint: {e}")
    _training_state["session"] = session
    return _session_summary(session)


# Serve built frontend if present.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    port = int(os.environ.get("MATH_SERVER_PORT", "5041"))
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(here)],
        log_level="info",
    )

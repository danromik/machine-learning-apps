"""FastAPI backend for the Image Classifier.

Endpoints:
  GET  /api/device                       current torch device
  GET  /api/device/list                  cpu / mps / cuda + hardware info
  POST /api/device/select                switch the active device

  GET  /api/dataset/classes              Imagenette class table
  GET  /api/dataset/status               on-disk state (downloaded? sizes?)
  POST /api/dataset/download             pull + extract tarball (synchronous)
  POST /api/dataset/download_stream      same, with progress events (NDJSON)
  POST /api/dataset/sample               training/val batch as base64 PNGs

  GET  /api/architecture/presets         LeNet-5 / AlexNet / ResNet-18

  POST /api/training/init                build a fresh training session
  GET  /api/training/state               current session status
  POST /api/training/reset               drop the live session
  POST /api/training/predict             forward-only on a list of base64 PNGs
  POST /api/training/eval                forward-only loss + accuracy
  POST /api/training/train_batch         one forward + backward + step
  GET  /api/training/checkpoints         list checkpoint files
  POST /api/training/checkpoints/save    save current session
  POST /api/training/checkpoints/load    load session from a file
  POST /api/training/checkpoints/delete  remove a checkpoint

  POST /api/inference/predict            classify one user-provided image
"""
from __future__ import annotations

import base64
import json
import platform
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch  # noqa: E402

import ml as _ml  # noqa: E402
import dataset as _dataset  # noqa: E402
import training as _training  # noqa: E402

STATIC_DIR = _HERE / "frontend" / "dist"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = pick_device()

app = FastAPI(title="Image Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5042", "http://127.0.0.1:5042"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Device introspection ───────────────────────────────────────────────


def _sysctl(name: str) -> str | None:
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
        clock = _sysctl("hw.cpufrequency_max")
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
    name: str


@app.post("/api/device/select")
def post_device_select(req: DeviceSelectReq):
    global device
    if req.name == "mps" and not torch.backends.mps.is_available():
        raise HTTPException(400, "MPS not available on this system")
    if req.name == "cuda" and not torch.cuda.is_available():
        raise HTTPException(400, "CUDA not available on this system")
    if req.name not in ("cpu", "mps", "cuda"):
        raise HTTPException(400, f"unknown device: {req.name!r}")
    new_device = torch.device(req.name)
    if str(new_device) != str(device):
        device = new_device
        s = _training_state["session"]
        if s is not None:
            s.model.to(device)
            for st in s.optimizer.state.values():
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(device)
            s.device = device
    return {"current": str(device)}


# ── Dataset endpoints ──────────────────────────────────────────────────


@app.get("/api/dataset/classes")
def get_classes():
    return {"classes": _ml.list_classes(), "input_size": _ml.INPUT_SIZE}


@app.get("/api/dataset/status")
def get_dataset_status():
    return _dataset.dataset_status()


@app.post("/api/dataset/download")
def post_download():
    """Synchronous download — blocks until done. ~88 MB tarball, then
    extracts. Slow paths (large download, slow network) should use the
    streaming variant below."""
    try:
        status = _dataset.download_imagenette()
    except Exception as e:
        raise HTTPException(500, f"download failed: {e}")
    _dataset.reset_indices()
    return status


@app.post("/api/dataset/download_stream")
def post_download_stream():
    """Streaming NDJSON: emits {stage, downloaded, total, fraction} events
    during download/extract, then a final {done: true, status: ...} line.
    The frontend uses this to render a progress bar."""
    import queue as queue_mod
    import threading

    events: queue_mod.Queue = queue_mod.Queue()
    SENTINEL = object()

    def progress(stage: str, downloaded: int, total: int) -> None:
        events.put(
            {
                "stage": stage,
                "downloaded": int(downloaded),
                "total": int(total),
                "fraction": (downloaded / total) if total > 0 else 0.0,
            }
        )

    def runner():
        try:
            status = _dataset.download_imagenette(progress_cb=progress)
            _dataset.reset_indices()
            events.put({"done": True, "status": status})
        except Exception as e:
            events.put({"error": str(e)})
        finally:
            events.put(SENTINEL)

    threading.Thread(target=runner, daemon=True).start()

    def stream():
        while True:
            ev = events.get()
            if ev is SENTINEL:
                break
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


class SampleBatchReq(BaseModel):
    split: str  # 'train' | 'val'
    count: int = 64
    seed: int = 42
    flip: bool = True
    jitter: float = 0.0
    random_crop: bool = True


@app.post("/api/dataset/sample")
def post_sample(req: SampleBatchReq):
    if req.split not in ("train", "val"):
        raise HTTPException(400, "split must be 'train' or 'val'")
    if not _dataset.dataset_status()["extracted"]:
        raise HTTPException(409, "dataset not downloaded — call /api/dataset/download first")
    cfg = _dataset.BatchRequest(
        split=req.split,
        count=req.count,
        seed=req.seed,
        flip=req.flip,
        jitter=req.jitter,
        random_crop=req.random_crop,
    )
    return {"samples": _dataset.sample_batch(cfg)}


@app.post("/api/dataset/sample_stream")
def post_sample_stream(req: SampleBatchReq):
    """Streaming variant — one sample per NDJSON line as PIL renders them.
    Used by the Preview Data modal so the user sees images appear
    progressively rather than waiting for the full batch."""
    if req.split not in ("train", "val"):
        raise HTTPException(400, "split must be 'train' or 'val'")
    if not _dataset.dataset_status()["extracted"]:
        raise HTTPException(409, "dataset not downloaded")
    cfg = _dataset.BatchRequest(
        split=req.split,
        count=req.count,
        seed=req.seed,
        flip=req.flip,
        jitter=req.jitter,
        random_crop=req.random_crop,
    )

    def stream():
        for s in _dataset.sample_batch_iter(cfg):
            yield json.dumps(s) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Architecture presets ───────────────────────────────────────────────


@app.get("/api/architecture/presets")
def get_architecture_presets():
    return {"presets": _ml.list_architecture_presets()}


# ── Training state ─────────────────────────────────────────────────────


class TrainingHyperparameters(BaseModel):
    lr: float
    batch_size: int
    optimizer: str


class InitTrainingReq(BaseModel):
    architecture: list[dict]
    preset: str | None = None
    hyperparameters: TrainingHyperparameters
    classes: list[str]
    dataset_config: dict | None = None


class PredictReq(BaseModel):
    images: list[str]


class TrainBatchReq(BaseModel):
    images: list[str]
    labels: list[str]
    lr: float | None = None
    optimizer: str | None = None


class CheckpointReq(BaseModel):
    filename: str


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
        "preset": s.preset,
    }


@app.post("/api/training/init")
def post_training_init(req: InitTrainingReq):
    try:
        session = _training.TrainingSession(
            req.architecture,
            req.preset,
            req.hyperparameters.model_dump(),
            req.classes,
            device,
            dataset_config=req.dataset_config,
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
    _training_state["session"] = None
    return {"has_session": False}


@app.post("/api/training/predict")
def post_predict(req: PredictReq):
    s = _require_session()
    return {"predictions": s.predict(req.images)}


@app.post("/api/training/eval")
def post_eval(req: TrainBatchReq):
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
    return {
        **_session_summary(session),
        "layers": session.layers,
        "preset": session.preset,
        "classes": session.classes,
        "dataset_config": session.dataset_config,
        "loss_history": session.loss_history,
        "val_loss_history": session.val_loss_history,
    }


# ── Inference ──────────────────────────────────────────────────────────


@app.post("/api/inference/predict")
async def post_inference(file: UploadFile = File(...)):
    """Classify a user-uploaded image. The image goes through the same
    val-pipeline preprocessing as held-out batches (resize-shorter →
    center crop), then through the live model, returning the input PNG
    + top-K class probabilities. Lets the user see exactly what the
    model sees."""
    s = _training_state["session"]
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        png_b64 = _dataset.encode_image_for_inference(raw)
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    item: dict[str, Any] = {
        "input_png_b64": png_b64,
        "predicted_label": None,
        "confidence": None,
        "top_k": [],
        "has_session": s is not None,
    }
    if s is not None:
        probs = s.predict([png_b64])[0]
        TOP_K = 5
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:TOP_K]
        top_k = [
            {"label": s.classes[idx], "confidence": float(prob), "index": idx}
            for idx, prob in ranked
        ]
        item["top_k"] = top_k
        item["predicted_label"] = top_k[0]["label"]
        item["confidence"] = top_k[0]["confidence"]
    return item


class InferenceSampleReq(BaseModel):
    """Pick a held-out val image at random — convenience for users who
    don't have an image to upload but want to see the model classify
    something. Same pipeline as a real upload."""
    seed: int | None = None


@app.post("/api/inference/sample")
def post_inference_sample(req: InferenceSampleReq):
    import random as _random

    if not _dataset.dataset_status()["extracted"]:
        raise HTTPException(409, "dataset not downloaded")
    seed = req.seed if req.seed is not None else _random.randint(0, 2**31 - 1)
    cfg = _dataset.BatchRequest(split="val", count=1, seed=seed)
    samples = _dataset.sample_batch(cfg)
    if not samples:
        raise HTTPException(500, "could not sample a val image")
    sample = samples[0]
    s = _training_state["session"]
    item: dict[str, Any] = {
        "input_png_b64": sample["png_b64"],
        "true_label": sample["label"],
        "source": sample["source"],
        "predicted_label": None,
        "confidence": None,
        "top_k": [],
        "has_session": s is not None,
    }
    if s is not None:
        probs = s.predict([sample["png_b64"]])[0]
        TOP_K = 5
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:TOP_K]
        top_k = [
            {"label": s.classes[idx], "confidence": float(prob), "index": idx}
            for idx, prob in ranked
        ]
        item["top_k"] = top_k
        item["predicted_label"] = top_k[0]["label"]
        item["confidence"] = top_k[0]["confidence"]
    return item


# Serve built frontend if present.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))
    port = int(os.environ.get("IMAGE_SERVER_PORT", "5041"))
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        reload_dirs=[str(here)],
        log_level="info",
    )

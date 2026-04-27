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
import platform
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
    return {"current": str(device)}


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

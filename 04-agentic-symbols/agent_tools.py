"""MCP tool surface exposed to the ML Engineer agent.

These tools wrap the same backend functionality the user can drive
through the UI: configure data synthesis, design the model, train it,
manage checkpoints, switch device. The agent reads from
`agent_state.pipeline_state` and writes via `agent_state.apply_patch` so
every mutation is broadcast to the WebSocket subscribers (the browser),
keeping the UI in lockstep with whatever the agent is doing.

Long-running operations (training loops) are intentionally bounded: a
single tool call runs N batches and returns metrics. The agent then
decides whether to call again — that gives it natural narration points
("loss plateaued; trying lower LR") and keeps each tool turn short
enough that the user's chat stays responsive.

Wiring: server.py imports `BUILD_MCP_SERVER` and calls
`set_session_accessors(...)` once at module load to bind the live
TrainingSession getter/setter and the device getter/setter. This breaks
the otherwise-circular import (server imports tools, tools need session
state living in server).
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Callable

import torch
from claude_agent_sdk import create_sdk_mcp_server, tool

import ml as _ml
import synthesis as _synthesis
import training as _training
from agent_state import (
    apply_patch,
    broadcast_event,
    get_state,
    pipeline_state,
    replace_state,
)


# ── Session accessors (injected by server.py) ──────────────────────────

# server.py owns the actual `_training_state` dict and the `device` global.
# It hands us getters/setters at startup so we don't have to import server
# (which would create a cycle: server → agent_tools → server).
_get_session: Callable[[], _training.TrainingSession | None] | None = None
_set_session: Callable[[_training.TrainingSession | None], None] | None = None
_get_device: Callable[[], torch.device] | None = None
_set_device: Callable[[str], torch.device] | None = None


def set_session_accessors(
    *,
    get_session: Callable[[], _training.TrainingSession | None],
    set_session: Callable[[_training.TrainingSession | None], None],
    get_device: Callable[[], torch.device],
    set_device: Callable[[str], torch.device],
) -> None:
    """Wire up the session/device handles the tools mutate. Server.py
    calls this once at startup."""
    global _get_session, _set_session, _get_device, _set_device
    _get_session = get_session
    _set_session = set_session
    _get_device = get_device
    _set_device = set_device


# ── Result helpers ─────────────────────────────────────────────────────


def _ok(payload: Any) -> dict[str, Any]:
    """Wrap a return value as an MCP tool result. JSON-encodes structured
    payloads so the LLM sees a string it can reason about."""
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict[str, Any]:
    """Return a tool error — the agent loop continues, sees the message,
    and can try a different approach."""
    return {"content": [{"type": "text", "text": f"ERROR: {message}"}], "is_error": True}


def _require_session() -> _training.TrainingSession:
    if _get_session is None:
        raise RuntimeError("session accessor not initialized")
    s = _get_session()
    if s is None:
        raise ValueError("no training session — call init_session first")
    return s


# ── Synthesis batch helper ─────────────────────────────────────────────


def _draw_batch(*, split: str, count: int, seed: int) -> tuple[list[str], list[str]]:
    """Pull `count` (image, label) pairs from the synthesis pipeline,
    using the synthesis config currently in pipeline_state.

    `split` is "train" or "val": picks training fonts vs validation fonts
    from the fontUsage map. Returns (images_b64, labels)."""
    syn = pipeline_state["synthesis"]
    categories = [cid for cid, on in syn["selectedCategories"].items() if on]
    font_usage = syn["fontUsage"]
    augmentation = syn["augmentation"]

    training_fonts = [f for f, role in font_usage.items() if role == "train"]
    validation_fonts = [f for f, role in font_usage.items() if role == "val"]

    req = _synthesis.SynthesisRequest(
        categories=categories,
        training_fonts=training_fonts,
        validation_fonts=validation_fonts,
        augmentation=augmentation,
        split=split,
        count=count,
        seed=seed,
    )
    images: list[str] = []
    labels: list[str] = []
    for sample in _synthesis.synthesize_iter(req):
        images.append(sample["png_b64"])
        labels.append(sample["label"])
    return images, labels


def _classes_for_session() -> list[str]:
    """Flatten selected categories into the class table the model
    will predict over. Same logic as the frontend uses to set
    `init` payload's `classes` field."""
    out: list[str] = []
    for cid, on in pipeline_state["synthesis"]["selectedCategories"].items():
        if not on:
            continue
        cat = _ml.SYMBOL_CATEGORIES.get(cid)
        if cat is not None:
            out.extend(cat["symbols"])
    return out


# ── Pipeline state read ────────────────────────────────────────────────


@tool(
    "get_pipeline_state",
    "Snapshot of the current pipeline: synthesis config, architecture, "
    "hyperparameters, training prefs, and live training session status "
    "(class count, parameter count, step, last loss/accuracy if available).",
    {},
)
async def get_pipeline_state(args: dict[str, Any]) -> dict[str, Any]:
    state = get_state()
    s = _get_session() if _get_session else None
    session_info: dict[str, Any] = {"has_session": s is not None}
    if s is not None:
        session_info.update({
            "num_classes": s.num_classes,
            "param_count": s.param_count(),
            "step": s.step,
            "lr": s.lr,
            "batch_size": s.batch_size,
            "optimizer": s.optimizer_name,
            "device": str(s.device),
            "last_train_loss": s.loss_history[-1]["loss"] if s.loss_history else None,
            "last_val_loss": s.val_loss_history[-1]["loss"] if s.val_loss_history else None,
            "train_steps_logged": len(s.loss_history),
            "val_evals_logged": len(s.val_loss_history),
        })
    return _ok({"pipeline": state, "session": session_info})


@tool(
    "get_recent_loss",
    "Last N entries from the training and validation loss histories.",
    {"n": int},
)
async def get_recent_loss(args: dict[str, Any]) -> dict[str, Any]:
    s = _get_session() if _get_session else None
    if s is None:
        return _ok({"train": [], "val": []})
    n = max(1, min(500, int(args.get("n", 50))))
    return _ok({
        "train": s.loss_history[-n:],
        "val": s.val_loss_history[-n:],
    })


@tool(
    "list_symbol_categories",
    "All available symbol categories (digits, lowercase roman, etc.) "
    "with their symbol lists and counts.",
    {},
)
async def list_symbol_categories(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"categories": _ml.list_categories()})


@tool(
    "list_curated_fonts",
    "Curated math/science fonts. Each entry includes the family name, a "
    "human note, and an `installed: bool` indicating whether the font is "
    "actually present on this Mac (only installed fonts can be used).",
    {},
)
async def list_curated_fonts(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"fonts": _ml.list_curated_fonts()})


@tool(
    "list_synthesis_presets",
    "The three built-in synthesis presets — beginner, intermediate, "
    "advanced — with the categories, train/val font split, and "
    "augmentation each one applies. Use this to inform "
    "apply_synthesis_preset() or to suggest manual config to the user.",
    {},
)
async def list_synthesis_presets(args: dict[str, Any]) -> dict[str, Any]:
    out = []
    for name in ("beginner", "intermediate", "advanced"):
        try:
            out.append(_ml.get_preset(name))
        except ValueError:
            pass
    return _ok({"presets": out})


@tool(
    "list_checkpoints",
    "All saved .pt checkpoint files with size and modification time.",
    {},
)
async def list_checkpoints(args: dict[str, Any]) -> dict[str, Any]:
    return _ok({"files": _training.list_checkpoints()})


@tool(
    "list_devices",
    "Compute devices available on this machine (cpu, mps if Apple "
    "Silicon, cuda if NVIDIA), with hardware info and which one is "
    "currently selected.",
    {},
)
async def list_devices(args: dict[str, Any]) -> dict[str, Any]:
    if _get_device is None:
        return _err("device accessor not initialized")
    # We can't import server to reuse _list_devices() (cycle), so we
    # compute here. Same logic, simplified — full hardware metadata is
    # in the GET /api/device/list endpoint.
    devs: list[dict[str, Any]] = [{"name": "cpu", "available": True}]
    if torch.backends.mps.is_available():
        devs.append({"name": "mps", "available": True})
    if torch.cuda.is_available():
        devs.append({"name": "cuda", "available": True})
    return _ok({"current": str(_get_device()), "devices": devs})


# ── Synthesis mutations ────────────────────────────────────────────────


@tool(
    "set_symbol_categories",
    "Choose which symbol categories to train on. Pass the full list of "
    "category IDs to enable (replaces the current selection). Valid IDs: "
    "digits, lower_roman, upper_roman, punctuation, other_ascii, "
    "lower_greek, upper_greek, math.",
    {"categories": list},
)
async def set_symbol_categories(args: dict[str, Any]) -> dict[str, Any]:
    cats = list(args.get("categories") or [])
    valid = set(_ml.SYMBOL_CATEGORIES.keys())
    bad = [c for c in cats if c not in valid]
    if bad:
        return _err(f"unknown category ids: {bad}. valid: {sorted(valid)}")
    # Frontend stores selectedCategories as a complete dict[str, bool] over
    # ALL category ids — set the chosen ones True and the rest False so a
    # patch overwrites cleanly without leaving stale True entries.
    selection = {cid: (cid in cats) for cid in _ml.SYMBOL_CATEGORIES.keys()}
    await apply_patch(
        {"synthesis": {"selectedCategories": selection, "activePreset": None}},
        source="agent",
    )
    return _ok({"enabled": cats, "class_count": len(_classes_for_session())})


@tool(
    "set_font_usage",
    "Assign each font family to 'train', 'val', or 'off'. The argument "
    "is a dict mapping family name → role. Only families set to 'train' "
    "are used during training; only those set to 'val' are used for "
    "validation (held out from training). 'off' fonts are skipped "
    "entirely. Use list_curated_fonts() to see installed family names.",
    {"fontUsage": dict},
)
async def set_font_usage(args: dict[str, Any]) -> dict[str, Any]:
    usage = dict(args.get("fontUsage") or {})
    valid_roles = {"train", "val", "off"}
    for fam, role in usage.items():
        if role not in valid_roles:
            return _err(f"font {fam!r} has invalid role {role!r}; must be one of {sorted(valid_roles)}")
    await apply_patch(
        {"synthesis": {"fontUsage": usage, "activePreset": None}},
        source="agent",
    )
    train = sum(1 for r in usage.values() if r == "train")
    val = sum(1 for r in usage.values() if r == "val")
    return _ok({"train_fonts": train, "val_fonts": val})


@tool(
    "set_augmentation",
    "Configure data augmentation. `noise` enables Gaussian pixel noise "
    "with `max_level` 0–100 controlling sigma. `skew` enables random "
    "horizontal shear ±15°. Both default to disabled. Pass the full "
    "object — partial updates are not supported.",
    {"noise": dict, "skew": dict},
)
async def set_augmentation(args: dict[str, Any]) -> dict[str, Any]:
    noise = dict(args.get("noise") or {})
    skew = dict(args.get("skew") or {})
    noise.setdefault("enabled", False)
    noise.setdefault("max_level", 0)
    skew.setdefault("enabled", False)
    await apply_patch(
        {"synthesis": {"augmentation": {"noise": noise, "skew": skew}, "activePreset": None}},
        source="agent",
    )
    return _ok({"augmentation": {"noise": noise, "skew": skew}})


@tool(
    "apply_synthesis_preset",
    "Apply a built-in synthesis preset by name (beginner | intermediate "
    "| advanced). Replaces selected categories, font usage, and "
    "augmentation in one shot. Use list_synthesis_presets() to see what "
    "each one configures.",
    {"name": str},
)
async def apply_synthesis_preset(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("name", "")).strip().lower()
    try:
        preset = _ml.get_preset(name)
    except ValueError as e:
        return _err(str(e))
    # Translate the preset into the pipeline_state shape:
    #   - selectedCategories: dict[str, bool] over ALL category ids
    #   - fontUsage: dict[str, "off"|"train"|"val"] over all curated fonts,
    #     so a patch overwrites cleanly without leaving stale entries.
    selected = {
        cid: (cid in preset["categories"])
        for cid in _ml.SYMBOL_CATEGORIES.keys()
    }
    font_usage: dict[str, str] = {f["family"]: "off" for f in _ml.list_curated_fonts()}
    for f in preset["training_fonts"]:
        font_usage[f] = "train"
    for f in preset["validation_fonts"]:
        font_usage[f] = "val"
    await apply_patch(
        {
            "synthesis": {
                "selectedCategories": selected,
                "fontUsage": font_usage,
                "augmentation": preset["augmentation"],
                "activePreset": name,
            }
        },
        source="agent",
    )
    return _ok({
        "preset": name,
        "categories": preset["categories"],
        "train_fonts": len(preset["training_fonts"]),
        "validate_fonts": len(preset["validation_fonts"]),
    })


# ── Architecture mutations ─────────────────────────────────────────────


# Architecture spec is a list of layer dicts. Using `list` as the schema
# tells the SDK it's a JSON array; the tool body validates structure.
LAYER_TYPES = {"conv2d", "maxpool2d", "flatten", "linear", "relu", "dropout"}


@tool(
    "set_architecture",
    "Set the model architecture as a list of layer dicts. Each layer is "
    "{type: str, params: dict}. Supported types: conv2d (params: "
    "out_channels, kernel, padding?, stride?), maxpool2d (kernel, "
    "stride?), flatten (no params), linear (out_features), relu (no "
    "params), dropout (p). The final classifier head (Linear → "
    "num_classes) is appended automatically — your last user-defined "
    "layer should produce a 1-D shape.",
    {"layers": list},
)
async def set_architecture(args: dict[str, Any]) -> dict[str, Any]:
    layers = args.get("layers")
    if not isinstance(layers, list):
        return _err("layers must be a list")
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            return _err(f"layers[{i}] is not a dict")
        if layer.get("type") not in LAYER_TYPES:
            return _err(f"layers[{i}].type {layer.get('type')!r} not in {sorted(LAYER_TYPES)}")
    await apply_patch({"architecture": {"layers": layers}}, source="agent")
    return _ok({"layer_count": len(layers)})


@tool(
    "set_hyperparameters",
    "Set training hyperparameters. lr (float), batch_size (int), "
    "optimizer (one of: adam, adamw, sgd). If a session is already live, "
    "lr and optimizer are hot-swapped on the next train_n_batches call; "
    "batch_size takes effect on the call after that.",
    {"lr": float, "batch_size": int, "optimizer": str},
)
async def set_hyperparameters(args: dict[str, Any]) -> dict[str, Any]:
    try:
        lr = float(args["lr"])
        batch_size = int(args["batch_size"])
        optimizer = str(args["optimizer"]).lower()
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"invalid hyperparameters: {e}")
    if optimizer not in ("adam", "adamw", "sgd"):
        return _err(f"optimizer must be adam/adamw/sgd; got {optimizer!r}")
    if lr <= 0 or batch_size <= 0:
        return _err("lr and batch_size must be positive")
    await apply_patch(
        {"architecture": {"hyperparameters": {
            "lr": lr, "batch_size": batch_size, "optimizer": optimizer,
        }}},
        source="agent",
    )
    return _ok({"lr": lr, "batch_size": batch_size, "optimizer": optimizer})


# ── Training operations ────────────────────────────────────────────────


@tool(
    "init_session",
    "Build a fresh training session from the current pipeline state "
    "(architecture + hyperparameters + selected categories). Replaces "
    "any existing session — call this when you want to start over with "
    "new architecture or class set. Errors if the architecture is empty "
    "or no categories are selected.",
    {},
)
async def init_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None or _get_device is None:
        return _err("session accessors not initialized")
    layers = pipeline_state["architecture"]["layers"]
    hp = pipeline_state["architecture"]["hyperparameters"]
    classes = _classes_for_session()
    if not layers:
        return _err("architecture is empty — call set_architecture first")
    if not classes:
        return _err("no symbol categories selected — call set_symbol_categories first")
    syn_cfg = {
        "selectedCategories": pipeline_state["synthesis"]["selectedCategories"],
        "fontUsage": pipeline_state["synthesis"]["fontUsage"],
        "augmentation": pipeline_state["synthesis"]["augmentation"],
    }
    try:
        session = _training.TrainingSession(
            layers, hp, classes, _get_device(), synthesis_config=syn_cfg,
        )
    except ValueError as e:
        return _err(str(e))
    _set_session(session)
    # Tell the UI a fresh session exists so the Training tab clears its
    # loss curves / step counter and starts updating live as we train.
    broadcast_event({
        "type": "training_session",
        "source": "agent",
        "hasSession": True,
        "numClasses": session.num_classes,
        "paramCount": session.param_count(),
        "step": session.step,
        "lossHistory": [],
        "valLossHistory": [],
    })
    return _ok({
        "num_classes": session.num_classes,
        "param_count": session.param_count(),
        "device": str(session.device),
    })


@tool(
    "reset_session",
    "Drop the live training session — model weights and optimizer state "
    "are discarded. Loss history charts in the UI are cleared. Use when "
    "you want to scrap progress and start fresh without changing the "
    "architecture.",
    {},
)
async def reset_session(args: dict[str, Any]) -> dict[str, Any]:
    if _set_session is None:
        return _err("session accessors not initialized")
    _set_session(None)
    broadcast_event({
        "type": "training_session",
        "source": "agent",
        "hasSession": False,
    })
    return _ok({"has_session": False})


@tool(
    "train_n_batches",
    "Run N training batches synchronously. Each batch synthesizes "
    "batch_size samples from the current synthesis config and runs one "
    "gradient step. Returns mean loss + mean accuracy across the N "
    "batches plus the final step counter. Bounded so the agent stays "
    "responsive — call again to keep training. n max 200 per call.",
    {"n": int},
)
async def train_n_batches(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    n = max(1, min(200, int(args.get("n", 1))))
    batch_size = int(pipeline_state["architecture"]["hyperparameters"]["batch_size"])
    losses: list[float] = []
    accs: list[float] = []
    rng_seed_base = random.randrange(2**31)
    for i in range(n):
        images, labels = _draw_batch(split="train", count=batch_size, seed=rng_seed_base + i)
        if not images:
            return _err(
                "synthesis returned no samples — check that selected categories "
                "have symbols supported by the chosen training fonts"
            )
        # Filter to labels the session knows about. (Agent might have changed
        # categories without re-init; surface that as an error rather than
        # silently dropping.)
        unknown = [lab for lab in labels if lab not in s.label_to_index]
        if unknown:
            return _err(
                f"batch contains labels not in current session class set: "
                f"{sorted(set(unknown))[:5]}{'…' if len(set(unknown)) > 5 else ''}. "
                "The session is built against an older class table — call "
                "init_session to rebuild against the current synthesis config."
            )
        try:
            result = s.train_batch(images, labels)
        except ValueError as e:
            return _err(str(e))
        losses.append(result["loss"])
        accs.append(result["accuracy"])
        # Stream a tick to the UI per batch so loss charts + step counter
        # update live during long training runs (otherwise the user just
        # sees the chart jump once when the tool returns N batches later).
        broadcast_event({
            "type": "training_tick",
            "source": "agent",
            "step": result["step"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
        })
        # Yield to the event loop so the WS sender task can flush the
        # queued tick before we block on the next batch's forward pass.
        await asyncio.sleep(0)
    return _ok({
        "batches_run": n,
        "mean_loss": sum(losses) / len(losses),
        "mean_accuracy": sum(accs) / len(accs),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "step": s.step,
    })


@tool(
    "eval_on_val",
    "Forward-only evaluation on a freshly-synthesized validation batch "
    "(uses the held-out validation fonts). Returns loss + accuracy. "
    "Doesn't change weights. Default count is 200.",
    {"count": int},
)
async def eval_on_val(args: dict[str, Any]) -> dict[str, Any]:
    try:
        s = _require_session()
    except (ValueError, RuntimeError) as e:
        return _err(str(e))
    count = max(1, min(2000, int(args.get("count", 200))))
    images, labels = _draw_batch(split="val", count=count, seed=random.randrange(2**31))
    if not images:
        return _err(
            "validation synthesis returned no samples — check that fonts "
            "marked 'validate' in fontUsage cover the selected categories"
        )
    # Drop unknown labels (model can't predict classes it never trained on);
    # report how many were dropped so the agent can react.
    keep = [(im, lab) for im, lab in zip(images, labels) if lab in s.label_to_index]
    dropped = len(images) - len(keep)
    if not keep:
        return _err("no validation samples have labels in the session's class set")
    images = [im for im, _ in keep]
    labels = [lab for _, lab in keep]
    try:
        result = s.eval_batch(images, labels)
    except ValueError as e:
        return _err(str(e))
    # Mirror the user-driven `maybeRunValidation` path: append a point
    # to the validation curve in the UI at the current training step.
    broadcast_event({
        "type": "validation_tick",
        "source": "agent",
        "step": s.step,
        "loss": result["loss"],
        "accuracy": result["accuracy"],
    })
    return _ok({
        "loss": result["loss"],
        "accuracy": result["accuracy"],
        "count": len(images),
        "dropped_unknown_labels": dropped,
    })


# ── Checkpoint operations ──────────────────────────────────────────────


@tool(
    "save_checkpoint",
    "Save the live training session to a .pt file under checkpoints/. "
    "Persists model weights, architecture, hyperparameters, class list, "
    "synthesis config, and loss/val curves — enough to fully restore the "
    "pipeline later. Filename is sandboxed to checkpoints/ (no path "
    "components).",
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
    if "/" in filename or "\\" in filename or ".." in filename:
        return _err("filename cannot contain path separators or '..'")
    name = _training.save_checkpoint(s, filename)
    return _ok({"name": name})


@tool(
    "load_checkpoint",
    "Load a checkpoint by filename. Replaces the live session AND "
    "restores synthesis + architecture + hyperparameter state in the UI "
    "(via state broadcast) so the agent and the user see the same "
    "configuration the model was trained against.",
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
    # Restore the pipeline state from the checkpoint so the UI reflects
    # what the model was trained against. Architecture + hyperparams are
    # in the session; synthesis config is in session.synthesis_config.
    new_pipeline = {
        "synthesis": session.synthesis_config or pipeline_state["synthesis"],
        "architecture": {
            "layers": session.layers,
            "hyperparameters": {
                "lr": session.lr,
                "batch_size": session.batch_size,
                "optimizer": session.optimizer_name,
            },
        },
        "training": pipeline_state["training"],  # training prefs aren't in ckpt
    }
    await replace_state(new_pipeline, source="agent")
    # The pipeline_state mirror doesn't carry session metadata or loss
    # series — those live in `training` $state on the frontend. Push a
    # session-restore event so the Training tab's step counter, param
    # count, and loss curves match the loaded model.
    broadcast_event({
        "type": "training_session",
        "source": "agent",
        "hasSession": True,
        "numClasses": session.num_classes,
        "paramCount": session.param_count(),
        "step": session.step,
        "lossHistory": list(session.loss_history),
        "valLossHistory": list(session.val_loss_history),
    })
    return _ok({
        "name": filename,
        "num_classes": session.num_classes,
        "param_count": session.param_count(),
        "step": session.step,
        "train_history_points": len(session.loss_history),
        "val_history_points": len(session.val_loss_history),
    })


@tool(
    "delete_checkpoint",
    "Remove a checkpoint file from disk. Filename sandboxed to the "
    "checkpoints/ directory.",
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
    "Switch the compute device to 'cpu', 'mps' (Apple Silicon GPU), or "
    "'cuda' (NVIDIA). If a session is live, the model and optimizer "
    "tensor state are migrated to the new device.",
    {"name": str},
)
async def select_device(args: dict[str, Any]) -> dict[str, Any]:
    if _set_device is None:
        return _err("device accessor not initialized")
    name = str(args.get("name", "")).strip().lower()
    if name not in ("cpu", "mps", "cuda"):
        return _err(f"name must be cpu/mps/cuda; got {name!r}")
    if name == "mps" and not torch.backends.mps.is_available():
        return _err("mps not available on this system")
    if name == "cuda" and not torch.cuda.is_available():
        return _err("cuda not available on this system")
    new_device = _set_device(name)
    return _ok({"current": str(new_device)})


# ── Build the MCP server ───────────────────────────────────────────────


ALL_TOOLS = [
    # read
    get_pipeline_state,
    get_recent_loss,
    list_symbol_categories,
    list_curated_fonts,
    list_synthesis_presets,
    list_checkpoints,
    list_devices,
    # synthesis
    set_symbol_categories,
    set_font_usage,
    set_augmentation,
    apply_synthesis_preset,
    # architecture
    set_architecture,
    set_hyperparameters,
    # training
    init_session,
    reset_session,
    train_n_batches,
    eval_on_val,
    # checkpoints
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
    # device
    select_device,
]


def build_mcp_server():
    """Construct the in-process MCP server with all tools registered."""
    return create_sdk_mcp_server(
        name="agentic-symbols",
        version="0.1.0",
        tools=ALL_TOOLS,
    )


def allowed_tool_names() -> list[str]:
    """The list of fully-qualified tool names the agent is allowed to
    call. SDK tool names get the prefix `mcp__<server-name>__<tool-name>`.
    Pass this to ClaudeAgentOptions.allowed_tools so the agent can only
    use our tools (no Read/Write/Bash etc.)."""
    return [f"mcp__agentic-symbols__{t.name if hasattr(t, 'name') else t.__name__}" for t in ALL_TOOLS]

# ML Learning Suite

A suite of interactive apps for learning and experimenting with machine learning concepts, each one a self-contained project that trains a real network end-to-end. The goal is pedagogical: each app should make a specific ML idea visible and manipulable (live loss curves, freehand inference, hyperparameter sliders, data inspection), and the sequence of projects should progress from simple MLPs to models that push the limits of the local hardware.

**Source:** https://github.com/danromik/machine-learning-apps (origin/main).

## Hardware & stack

- Apple Silicon Mac, PyTorch on the **MPS** backend (`torch.device("mps")`); CUDA/CPU paths kept as fallbacks.
- Python 3.13, managed with **uv**.
- **One shared virtual env at the repo root** (`.venv/` + `pyproject.toml`). Adding a dep: `uv add <pkg>` from the root. Don't create per-project venvs — PyTorch is ~2GB and reinstalling it per project is painful. Projects that later need incompatible deps can split out.
- Frontend uses **Svelte 5 + Vite + Tailwind**, served by FastAPI as static `dist/` in production and proxied through the Vite dev server in development.

## Repo layout

```
ml/
├── pyproject.toml             # shared Python deps
├── .venv/                     # shared env
├── CLAUDE.md
├── 01-mnist-trainer/          # canonical reference project (FastAPI + Svelte)
├── 02-math-symbols/           # printed-symbol OCR over synthesized data
└── NN-<next-project>/         # future
```

Each learning project lives in its own `NN-<name>/` subfolder, numbered in curriculum order. The numbering is load-bearing — it's the progression, not just ordering. **Each project is self-contained** (its own data/, checkpoints/, models, training loop) so it can be deleted, copied, or forked without breaking siblings.

## Project shape (from `01-mnist-trainer`)

Every project should follow this layout:

```
NN-name/
├── models.py            # nn.Module classes, build_model, pick_device, get_loaders, DATA_DIR, CKPT_DIR
├── ml.py                # app-specific helpers: sampling, prediction from canvas, architecture description, checkpoint I/O
├── training_worker.py   # background-thread training loop; emits events onto a queue.Queue
├── train.py             # CLI entry: `uv run python NN-name/train.py --model cnn --epochs 5`
├── server.py            # FastAPI app: REST endpoints + WebSocket that streams worker events
├── run.sh               # build frontend + start backend + open browser
├── data/                # dataset cache (created on first run)
├── checkpoints/         # saved .pt files
└── frontend/            # Svelte 5 + Vite + Tailwind app
    ├── src/
    │   ├── App.svelte
    │   ├── api.ts
    │   ├── state.svelte.ts   # global $state stores (cfg, ui, explorer, chartData)
    │   ├── components/       # NeuralNetworkCard, DrawClassify, DataExplorer, LossChart, ValChart, …
    │   └── …
    └── dist/                 # built output served by FastAPI
```

`models.py` owns the path constants — everything else imports `DATA_DIR`/`CKPT_DIR` from it so there's a single source of truth.

## Backend conventions

These describe `01-mnist-trainer`'s pattern. `02-math-symbols` deviates — see its section below.

- **Threaded training worker** (`training_worker.py`) with a `queue.Queue` of event dicts (`reset`, `log`, `start`, `step`, `epoch`, `checkpoint`, `paused`, `stopped`, `done`, `error`). The FastAPI WebSocket endpoint drains the queue and pushes events as JSON. Training never blocks the HTTP server.
- **Persistent training session** between button clicks so "single batch", "single epoch", and "continuous" all advance the same model. Session auto-rebuilds when incompatible config changes (model type, batch size); LR is hot-swappable. `start_run` accepts `max_steps` and `max_epochs` caps; whichever hits first ends the run.
- **Predict path** (`/api/predict`) prefers, in order: live training session → cached last-loaded model → first checkpoint on disk. After `reset_session(cfg)`, the cached model is cleared so prediction always uses the freshly initialized random weights.
- **Ports**: backend serves on `5041` (the URL the user opens — mnemonic: first four MNIST training labels are 5,0,4,1). Vite dev server runs on `5042` and proxies `/api` and `/ws` back to `5041`. Override with `MNIST_SERVER_PORT`.

## Frontend conventions

- **Svelte 5 runes** (`$state`, `$derived`, `$effect`) — no stores. Global state lives in `state.svelte.ts`.
- **Cross-component coordination via shared state**, not events. e.g., clicking a sample in `DataExplorer` sets `ui.loadImage = { png_b64, label, seq }` (the `seq` counter forces reactivity even when the same image is re-clicked); `DrawClassify` watches it via `$effect`.
- **WebSocket events** are typed in `api.ts` (`TrainEvent` discriminated union) and reduced in `App.svelte`'s `handleEvent` switch.
- **Virtualized lists** for anything that could exceed a few hundred items. Data Explorer uses absolute-positioned cells in a tall scroll container, sized to the full filtered count, fetching 500-item chunks on demand.
- **Tailwind via CSS variables** (`--color-accent`, `--color-muted`, etc.) so themes can be swapped centrally.

## Commands

```bash
# run the web app (build frontend, start backend, open browser to http://localhost:5041)
01-mnist-trainer/run.sh

# CLI training (ground truth, fastest to iterate)
uv run python 01-mnist-trainer/train.py --model cnn --epochs 5

# add a new dependency (always from repo root)
uv add <package>

# frontend dev with HMR (separate terminal; backend must already be running)
cd 01-mnist-trainer/frontend && npm run dev    # vite on :5042, proxies to backend on :5041
```

## Math Symbol Trainer (`02-math-symbols/`)

A printed-symbol OCR classifier covering the glyphs used in mathematical writing: alphanumerics, punctuation/ASCII, Greek upper/lower, and math symbols (several hundred classes total). Distinct from `01-mnist-trainer` in three ways:

1. **Synthesized data, not downloaded.** Glyphs are rendered from system/open-source fonts on the fly during training. No fixed dataset on disk; each training step pulls a fresh batch with augmentation (noise, blur, skew, etc.). On-the-fly synthesis means infinite training data and lets augmentation be a live, tunable hyperparameter. Missing-glyph `(char, font)` pairs are filtered out at sample time so the model never sees a `.notdef` placeholder mislabeled as some target symbol.
2. **Held-out fonts for validation.** Generalization to *unseen fonts* is the bar — a fixed seeded validation set is rendered with fonts the training pipeline never sees, so val accuracy measures real generalization rather than memorization of font-class pairs.
3. **Tabbed pedagogical UI.** Numbered tabs walk the user through the pipeline:
   - **0 Orientation** — welcome page describing the pipeline, with a CTA into the Data Synthesis tab.
   - **1 Data Synthesis** — choose symbol categories (digits, roman, punctuation, Greek, math), training fonts, held-out validation fonts, and augmentation. Includes a Preview Data modal that streams sample images for both splits.
   - **2 Model Architecture** — design the network layer-by-layer (drag-and-drop) with a Suggest button that picks a CNN matched to the current data config.
   - **3 Training** — Re-Initialize Model (capsule CTA, distinct from train buttons), Train 1 Batch (Fun) which sweeps a per-image highlight + per-class softmax bars + green/red verdict frames over the batch, Train 1 Batch (Fast) which skips the animation, Train 1 Epoch (one epoch = enough batches for each symbol to be seen on average `samples_per_symbol_per_epoch` times — user-tunable), Train Continuously (with Stop). Right pane stacks two row-equal sections: top is the per-sample probability chart + a persistent Batch chart of green/red verdict counts that survives across batch reloads; bottom is two `LossChart`s — training loss (one point per gradient step) and validation loss (sparser; runs every `validateEveryN` steps via `/api/training/eval` on a freshly-rendered held-out batch). Hyperparameters (LR, optimizer) are hot-swappable: the slider value is sent with every train_batch request and the backend updates the optimizer in place.
   - **4 Inference** — single-line input field accepts plain text with `$…$` math delimiters; KaTeX renders the preview inline (only the math regions go through KaTeX, plain text stays plain). Glyphs are extracted by walking the rendered DOM (so `\alpha` correctly expands to `α`), then `/api/inference/render` runs each through the synthesis pipeline + the live model in one round-trip, returning input PNGs plus the **top-K** predictions per cell (top-1 + 4 alternatives, each with its rendered glyph and confidence). Layout: a 2×2 grid with predicted glyphs grid (each cell shows the model's top-1 with a confidence progress bar) + predicted text in the left column, and a per-glyph **detail panel** spanning both rows in the right column. Hovering a predicted cell highlights it and populates the panel with the input glyph, top-1 confidence, and the top-K alternatives as a confidence-bar table — green bar for the input class when it's in top-K, red for an incorrect top-1. Manual "Run Inference" button — not auto-debounced.

### Backend specifics (differs from MNIST)

- **No worker thread, no WebSocket.** All training endpoints are synchronous REST: `POST /api/training/train_batch` runs one gradient step and returns. Sequencing batches into "1 epoch" or "continuously" happens client-side (a JS loop in `TrainingTab.svelte` that calls train_batch + loadBatch repeatedly, with a `Stop` flag to abort).
- **Files.** `ml.py` (symbol catalog + curated font list + presets), `synthesis.py` (glyph rendering + augmentation), `training.py` (TrainingSession + checkpoint I/O), `server.py` (FastAPI). No `training_worker.py`, no `train.py` CLI yet.
- **Hot-swap hyperparameters via train_batch kwargs.** `lr` and `optimizer_name` can be passed with each train_batch request; the session updates the optimizer in place (rebuilds on optimizer-name change, preserves Adam state on lr-only change). `save_checkpoint` reads from live session attrs, not the stale init-time `hyperparameters` dict.
- **Endpoints beyond the obvious CRUD.** `POST /api/training/eval` runs forward-only loss + accuracy on a held-out batch (used by the validation curve). `POST /api/training/reset` drops the live session — fired by the frontend whenever Data Synthesis settings change, since the model's class table is tied to that config. `GET /api/device/list` returns CPU + MPS (+ CUDA if present) with hardware info (cores, memory, clock); `POST /api/device/select` updates the global device and, if a session exists, moves both the model and the optimizer's tensor state (Adam moments etc., which `model.to()` doesn't follow) to the new device. `POST /api/inference/render` does the full inference pipeline in one call: render each char, batch-predict, return top-K rendered alternatives + confidences per cell.
- **Synthesis-change invalidates everything.** An `$effect` in `App.svelte` watches a JSON signature of `synthesis.{selectedCategories, fontUsage, augmentation}`; on change it clears `architecture.layers`, all `training.*` state (including `batch`, `lossHistory`, `valLossHistory`, `batchChartCounts`), and POSTs `/api/training/reset` so the backend session is dropped. Keeps the model from being trained against one class set then evaluated against another.
- **Tab status subtitles** in `App.svelte` reactively show `X symbols · Y fonts` (Data Synthesis), `X layers · Y weights` (Architecture), `X epochs · Y batches · Z% accuracy` (Training) — derived from state and updated live during training runs.
- **Don't regenerate the batch on tab remount.** `training.batch` lives in global state, so it survives unmount/remount. `TrainingTab`'s loadBatch effect tracks a `lastBatchConfigSig` per component instance: on first run after mount it skips the load if a batch is already populated; later runs only fetch when the synthesis config signature actually changed. This avoids burning a fresh batch (and dropping verdict frames + selection) every time the user switches tabs.

**Build order.** Built incrementally tab by tab. All five tabs are now live; CLI (`train.py`) is the main remaining piece if needed for headless/scriptable runs.

**Ports.** Same as MNIST: backend on `5041`, Vite dev on `5042`. Override with `MATH_SERVER_PORT`. Only one app can run at a time without overriding.

**Frontend deps beyond the shared stack:** `katex` (loaded globally via `main.ts`) for the Inference tab.

## Adding a new project

1. Create `NN-<name>/` at the repo root (e.g., `03-cifar10/`). Copy `01-mnist-trainer/` as a template if the shape fits.
2. If new Python deps are needed, `uv add` them from root.
3. Self-contained — no cross-project imports. Each project owns its `models.py`, `data/`, `checkpoints/`.
4. Keep the CLI (`train.py`) and the GUI (`server.py` + `frontend/`) as separate entry points — the CLI is the ground truth (fastest to iterate, scriptable), the GUI is the teaching surface.

## Design priorities (in order)

1. **Make the ML concept visible.** If a feature doesn't help the user see or feel what the model is doing, skip it.
2. **Keep training feedback live.** Loss curves and metrics update during training, not just at the end.
3. **Preserve the CLI.** Every GUI project has a runnable CLI equivalent.
4. **One concept per project.** Don't combine lessons; the curriculum does that.

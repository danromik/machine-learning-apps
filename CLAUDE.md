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
├── 03-image-classifier/       # natural-image classifier on Imagenette (WIP)
├── 04-agentic-symbols/        # Math Symbols pipeline + embedded ML Engineer agent
├── 05-agentic-snake/          # Reinforcement learning: Snake + embedded RL Coach agent
├── 06-agentic-cube/           # RL on a Rubik's Cube: value iteration + reverse-scramble curriculum + RL Coach
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
3. **Tabbed pedagogical UI.** Numbered tabs walk the user through the pipeline. Active tab is persisted to localStorage so a reload returns the user to where they left off. Tab style: rounded-frame around the active tab (no underline), with a soft accent-tinted gradient and drop shadow.
   - **0 Orientation** — welcome page describing the pipeline, with a CTA into the Data Synthesis tab.
   - **1 Data Synthesis** — choose symbol categories (digits, roman, punctuation, Greek, math), training fonts, held-out validation fonts, and augmentation. Includes a Preview Data modal that streams sample images for both splits.
   - **2 Model Architecture** — design the network layer-by-layer (drag-and-drop) with a Suggest button that picks a CNN matched to the current data config.
   - **3 Training** — capsule CTA whose label flips between **Initialize Model** (no live session) and **Re-Initialize Model** (replacing an existing one); Train 1 Batch (Fun) which sweeps a per-image highlight + per-class softmax bars + green/red verdict frames over the batch; Train 1 Batch (Fast) which skips the animation; Train 1 Epoch (one epoch = enough batches for each symbol to be seen on average `samples_per_symbol_per_epoch` times — user-tunable); Train Continuously (with Stop). Right pane stacks two row-equal sections: top is the per-sample probability chart + a persistent Batch chart of green/red verdict counts that survives across batch reloads; bottom is two `LossChart`s — training loss (one point per gradient step) and validation loss (sparser; runs every `validateEveryN` steps via `/api/training/eval` on a freshly-rendered held-out batch). Hyperparameters (LR, optimizer) are hot-swappable: the slider value is sent with every train_batch request and the backend updates the optimizer in place. Sidebar's Checkpoints section also shows: a "filesize / last save" line under the filename input (data sourced from the same metadata fetch that powers the file table), Auto-save and Auto-load on restart toggles, and a **Saved files** list with one row per `.pt` file — clicking a name populates the input, the X button on each row deletes the file. Filename, both toggles, and active tab all persist across browser reloads via localStorage.
   - **4 Inference** — single-line input field accepts plain text with `$…$` math delimiters; KaTeX renders the preview inline (only the math regions go through KaTeX, plain text stays plain). Glyphs are extracted by walking the rendered DOM (so `\alpha` correctly expands to `α`), then `/api/inference/render` runs each through the synthesis pipeline + the live model in one round-trip, returning input PNGs plus the **top-K** predictions per cell (top-1 + 4 alternatives, each with its rendered glyph and confidence). Layout: a 2×2 grid with predicted glyphs grid (each cell shows the model's top-1 with a confidence progress bar) + predicted text in the left column, and a per-glyph **detail panel** spanning both rows in the right column. Hovering a predicted cell highlights it and populates the panel with the input glyph, top-1 confidence, and the top-K alternatives as a confidence-bar table — green bar for the input class when it's in top-K, red for an incorrect top-1. Manual "Run Inference" button — not auto-debounced.
   - **5 Debrief** — contextual summary of the current pipeline state (synthesis: symbols / training fonts / val fonts / augmentation; architecture: layers, params, lr, batch size, optimizer; training: steps, epochs, last batch loss + accuracy, class count) plus a plain-text message that adapts to one of nine status buckets — `loading`, `no-symbols`, `no-architecture`, `no-training`, `just-loaded` (checkpoint loaded but no fresh training in this session yet), `early-training`, `progressing`, `almost-there`, `fully-trained`. The "great" accuracy threshold scales with pipeline difficulty (easy <30 symbols, medium 30–150, advanced >150) so the congratulations message fires at a sensible bar for the chosen task. Inline links navigate the user to the relevant tab to take the next step.

### Backend specifics (differs from MNIST)

- **No worker thread, no WebSocket.** All training endpoints are synchronous REST: `POST /api/training/train_batch` runs one gradient step and returns. Sequencing batches into "1 epoch" or "continuously" happens client-side (a JS loop in `TrainingTab.svelte` that calls train_batch + loadBatch repeatedly, with a `Stop` flag to abort).
- **Files.** `ml.py` (symbol catalog + curated font list + presets), `synthesis.py` (glyph rendering + augmentation), `training.py` (TrainingSession + checkpoint I/O), `server.py` (FastAPI). No `training_worker.py`, no `train.py` CLI yet.
- **Hot-swap hyperparameters via train_batch kwargs.** `lr` and `optimizer_name` can be passed with each train_batch request; the session updates the optimizer in place (rebuilds on optimizer-name change, preserves Adam state on lr-only change). `save_checkpoint` reads from live session attrs, not the stale init-time `hyperparameters` dict.
- **Checkpoints persist the full pipeline, not just weights.** `save_checkpoint` writes `state_dict`, `layers` (architecture spec), `hyperparameters` (lr / batch_size / optimizer), `classes` (output class table), `step`, `synthesis_config` (`{selectedCategories, fontUsage, augmentation}` snapshot), and the live `loss_history` / `val_loss_history` series (capped at 2000 points each, mirroring the frontend cap). `load_checkpoint` reconstructs a `TrainingSession` from all of this. `/api/training/checkpoints/load` returns everything in one payload so the frontend's `applyCheckpointResponse` can rewrite synthesis state, the architecture diagram, hyperparameter sliders, and both loss charts in one go — under a `training.suppressSynthesisInvalidation` counter so the synthesis-change effect's wholesale reset doesn't fire mid-restore.
- **Endpoints beyond the obvious CRUD.** `POST /api/training/eval` runs forward-only loss + accuracy on a held-out batch (used by the validation curve). `POST /api/training/reset` drops the live session — fired by the frontend whenever Data Synthesis settings change, since the model's class table is tied to that config. `GET /api/training/checkpoints` returns one dict per file (`name`, `size`, `mtime`) so the sidebar can render the saved-files list with tooltip metadata. `POST /api/training/checkpoints/delete` removes a checkpoint; the filename is sandboxed (rejects `/`, `\`, `..`) so the endpoint can't traverse out of `checkpoints/`. `GET /api/device/list` returns CPU + MPS (+ CUDA if present) with hardware info (cores, memory, clock); `POST /api/device/select` updates the global device and, if a session exists, moves both the model and the optimizer's tensor state (Adam moments etc., which `model.to()` doesn't follow) to the new device. `POST /api/inference/render` does the full inference pipeline in one call: render each char, batch-predict, return top-K rendered alternatives + confidences per cell.
- **Synthesis-change invalidates everything.** An `$effect` in `App.svelte` watches a JSON signature of `synthesis.{selectedCategories, fontUsage, augmentation}`; on change it clears `architecture.layers`, all `training.*` state (including `batch`, `lossHistory`, `valLossHistory`, `batchChartCounts`), and POSTs `/api/training/reset` so the backend session is dropped. Keeps the model from being trained against one class set then evaluated against another.
- **Tab status subtitles** in `App.svelte` reactively show `X symbols · Y fonts` (Data Synthesis), `X layers · Y weights` (Architecture), `X epochs · Y batches · Z% accuracy` (Training), and a status hint for Debrief (`design needed` → `ready to train` → `XX.X% accuracy`) — derived from state and updated live during training runs.
- **Don't regenerate the batch on tab remount.** `training.batch` lives in global state, so it survives unmount/remount. `TrainingTab`'s loadBatch effect tracks a `lastBatchConfigSig` per component instance: on first run after mount it skips the load if a batch is already populated; later runs only fetch when the synthesis config signature actually changed. This avoids burning a fresh batch (and dropping verdict frames + selection) every time the user switches tabs.
- **Run-state lives on the global `training` store.** `busy`, `statusMsg`, `epochRunning`, `abortEpoch`, `continuousRunning`, `abortContinuous` are all on the global `training` $state object — *not* component-local in `TrainingTab.svelte`. Reason: the train-1-epoch / train-continuously loops are JS closures that survive unmount (they reference module-level state), but if the UI flags lived on the component then a navigate-away-and-back would reset Stop buttons and the status footer to "idle" while the loop kept running. Lifting these to the same store the loop is mutating keeps the buttons + footer in sync with the still-running loop after remount, and lets a Stop click from the new component instance set the abort flag the old loop's `while` reads.
- **Auto-save / auto-load on restart.** Auto-save fires after every train op via a `maybeAutoSave` helper called from each loop's `finally` block (so it covers clean exit, user-initiated stop, *and* thrown errors). Auto-load fires from a one-shot effect in `App.svelte` once `synthesis.loaded` is true, attempts to load the file in `training.checkpointFilename`, silently skips when no file matches. Filename and both toggles are persisted to localStorage (`math-symbols.checkpoint-prefs.v1`) on every change. Active tab is persisted under a separate key (`math-symbols.ui-prefs.v1`) by an `$effect` in `App.svelte` that calls `persistUiPrefs()` on every `ui.activeTab` change.

**Build order.** All six tabs are live; CLI (`train.py`) is the main remaining piece if needed for headless/scriptable runs.

**Ports.** Same as MNIST: backend on `5041`, Vite dev on `5042`. Override with `MATH_SERVER_PORT`. Only one app can run at a time without overriding.

**Frontend deps beyond the shared stack:** `katex` (loaded globally via `main.ts`) for the Inference tab.

## Image Classifier (`03-image-classifier/`) — work in progress

Photo-classification app modeled on the Math Symbol Trainer's six-tab pipeline but adapted for natural images. Trains on **Imagenette** — fast.ai's 10-class subset of ImageNet — with a fixed 3×96×96 input and a choice of three preset architectures spanning two decades of CNN history. Status: end-to-end functional (download → train → infer); copy + ergonomics still being polished; no `train.py` CLI yet.

### What's different from Math Symbols

1. **Real downloaded data, not synthesized.** First-run flow is one click in the Data Acquisition tab → `urllib` downloads `imagenette2-160.tgz` (~88 MB) → `tarfile` extracts into `data/` → an in-memory `ImagenetteIndex` walks `<split>/<wnid>/*.JPEG`. Augmentation is now a *training-pipeline* concern (random crop, horizontal flip, color jitter) rather than a synthesis-time choice; val images go through deterministic resize + center-crop only. Class set is fixed (10 Imagenette labels) so changing augmentation doesn't invalidate the session — only the in-memory batch is dropped.
2. **Three preset architectures, plus Custom.** `ml.py:ARCHITECTURE_PRESETS` holds `lenet5` (1998, sequential layer list), `alexnet` (2012, sequential layer list adapted to 96×96 with 1024-wide FC instead of 4096), and `resnet18` (2015, `locked: True`, built directly via `torchvision.models.resnet18(weights=None)` with the FC retargeted to 10 classes). LeNet and AlexNet apply onto the drag-and-drop canvas as editable layer lists; ResNet shows as a single placeholder block in the diagram and bypasses the layer-list builder entirely. Editing a preset clears `architecture.preset` so the selector no longer claims ownership of the canvas.
3. **`build_model` dispatches on `preset`.** `training.py:build_model(layers, preset, num_classes)` — when `preset == 'resnet18'`, calls `build_resnet18(num_classes)`; otherwise runs `build_layered_model(layers, num_classes)`, which is the same builder Math Symbols uses (conv2d/maxpool2d/flatten/linear/relu/dropout) plus `batchnorm2d`. The implicit final `nn.Linear → num_classes` head is only appended in the layered path; ResNet supplies its own.
4. **ImageNet normalization in the decode path.** `training.py:_decode_image` divides by 255, subtracts mean `[0.485, 0.456, 0.406]`, divides by std `[0.229, 0.224, 0.225]`, and reshapes HWC→CHW before stacking. ResNet-18 from torchvision was designed for these exact stats so it really matters there; layered architectures benefit from centered inputs anyway.
5. **Inference is image upload, not text + KaTeX.** `POST /api/inference/predict` accepts a `multipart/form-data` upload, runs it through the val pipeline (resize-shorter → center-crop → encode), classifies, and returns the input PNG + top-5 labels with confidences. `POST /api/inference/sample` pulls a random val image for users without an image to upload — same response shape, plus `true_label` so the UI can render a green ✓ / red ✗ verdict. The Inference tab in the frontend has a drag-drop zone and a "Sample a val image" button; no KaTeX dependency.
6. **State store is `dataset`, not `synthesis`.** `state.svelte.ts` has a `dataset` $state with `{ loaded, classes, status, augmentation, downloading, downloadStage, downloadFraction }`. `applyArchitecturePreset` writes `architecture.preset` and either replaces the layer list (editable presets) or empties it (locked). `INPUT_SHAPE` is `[3, 96, 96]`. The dataset-change effect only invalidates the *current batch* on augmentation change (class set is fixed) — it does *not* tear down the architecture or session like Math Symbols' synthesis-change effect does.
7. **Backend file layout.** `ml.py` (Imagenette catalog + presets), `dataset.py` (download + `BatchRequest` + `sample_batch[_iter]` + base64 PNG round-trip), `training.py` (TrainingSession with both layered + locked-preset paths), `server.py` (FastAPI). Mirrors Math Symbols structurally; `synthesis.py` is replaced by `dataset.py`.
8. **Preset state in checkpoints.** `save_checkpoint` writes `state_dict, layers, preset, hyperparameters, classes, step, dataset_config, loss_history, val_loss_history`. `load_checkpoint` reconstructs the `TrainingSession` via the same `build_model(layers, preset, num_classes)` dispatch. The frontend's `applyCheckpointResponse` sets `architecture.preset` from the response so a saved ResNet-18 reloads with the locked-preset placeholder block, not as an empty layer list.
9. **One epoch = `ceil(num_train / batch_size)` batches.** Imagenette has ~9469 train images, so a batch_size of 64 gives ~148 batches per epoch. No `samples_per_class_per_epoch` knob (Math Symbols needed it because synthesized data has no inherent epoch boundary; here the dataset is finite). Validation cadence (`validateEveryN`) survives as a tunable hyperparameter.

**Endpoints.** Beyond the shared `/api/device*` and `/api/training/*` shape: `GET /api/dataset/classes`, `GET /api/dataset/status`, `POST /api/dataset/download` (synchronous), `POST /api/dataset/download_stream` (NDJSON progress events from a background thread), `POST /api/dataset/sample[_stream]` (batch with augmentation), `GET /api/architecture/presets`, `POST /api/inference/predict` (multipart upload), `POST /api/inference/sample` (random val image).

**Ports.** Backend on `5041`, Vite dev on `5042`, override via `IMAGE_SERVER_PORT`.

**Extra Python deps.** `python-multipart` for the inference upload endpoint (added at repo root). `torchvision` was already in `pyproject.toml` from MNIST.

## Agentic Symbol Trainer (`04-agentic-symbols/`)

A re-imagining of `02-math-symbols/` where an embedded **ML Engineer agent** (Claude Opus 4.7, 1M context) drives the same pipeline the user can — categories, architecture, training, eval, checkpoints. The user sees the agent's tool calls in a chat sidebar on the right and the resulting state changes mirrored live in the tabbed UI on the left. The synthesis / architecture / training tabs are forked from Math Symbols; the new pieces are the agent runtime, the shared pipeline-state mirror, and the chat pane.

### Architecture differences vs Math Symbols

1. **Shared pipeline-state mirror (backend = source of truth).** `agent_state.py` owns `pipeline_state` (synthesis + architecture + training-prefs slices, mirroring frontend store shapes verbatim so no translation is needed in the WS layer). Both drivers read and write it: the UI via `POST /api/state/patch` (debounced 200ms), the agent via MCP tools that call `apply_patch(...)`. Every mutation broadcasts a `state_patch` (or `state_replace`) over `/ws/state` to all subscribers, tagged with `source: "ui" | "agent" | "system"` so the originator can ignore its own echo. Echo prevention on the frontend uses a `lastSyncedSig` JSON signature: when a WS event arrives we update the sig before applying the patch so the auto-sync `$effect` sees no diff and skips the round-trip.

2. **Agent runtime + MCP tool surface.** `agent_runtime.py` runs one user→assistant turn against the Claude Agent SDK and yields normalized SSE events (text deltas, tool calls, tool results, usage, final result). `agent_tools.py` defines ~20 in-process MCP tools in three layers: read (`get_pipeline_state`, `get_recent_loss`, `list_*`), mutating (`set_symbol_categories`, `set_architecture`, `set_hyperparameters`, `apply_synthesis_preset`), and training (`init_session`, `reset_session`, `train_n_batches`, `eval_on_val`, `save_checkpoint`, `load_checkpoint`, `select_device`). The agent's allowed-tools list is restricted to these — no Read/Write/Bash on the host. `train_n_batches` is bounded (max 200 per call) so each tool turn is short and the agent gets natural narration points.

3. **Live agent-driven training updates.** When the agent calls `train_n_batches`, the tool broadcasts a `training_tick` event over `/ws/state` after every gradient step (with an `await asyncio.sleep(0)` so the WS sender flushes before the next forward pass blocks the loop). Frontend's WS handler in `App.svelte` reduces these into the `training` store: `training_tick` pushes onto `lossHistory` and updates step / lastLoss / lastAccuracy; `validation_tick` pushes onto `valLossHistory`; `training_session` resets or restores the full session state (used by `init_session`, `reset_session`, `load_checkpoint`). Result: loss charts and the step counter on the Training tab update live during a long agent run rather than jumping once when the tool turn returns. These three event types ride the same WebSocket as `state_patch` / `state_replace` but bypass the deep-merge `pipeline_state` model — they're transient progress signals, not shared state.

4. **Chat pane on the right.** `ChatPane.svelte` composes a header bar (`UsageBar` — left-aligned "ML Engineer Chat" label + right-aligned 1M-token-context gauge), `SessionMenu` (resume past chats / new chat), `ChatTranscript`, and a Send/Stop composer. The transcript is reduced from SSE events: `text_delta` appends to the open assistant bubble, `text_message` closes it, `tool_use` opens a tool row (status=running) which flips to success/error when the matching `tool_result` arrives. Past sessions live in `~/.claude/projects/<encoded-cwd>/<id>.jsonl` — `GET /api/agent/sessions` and `/sessions/{id}/messages` read them back to reconstruct transcripts. Pane visibility, active session id, and pane width all persist to localStorage under `agentic-symbols.chat-prefs.v1`.

5. **Resizable chat pane.** A 4px-wide divider sibling renders before the `<aside>` (replacing the old `border-l`); pointerdown captures startX + startWidth, pointermove updates `chat.paneWidth` live (the aside's `style="width:"` and `<main class="flex-1 min-w-0">` reflow on every frame), pointerup persists. Listeners are on `window` so a drag past the chrome still releases cleanly; body cursor + `userSelect` are pinned across the drag. Bounds: min 240px, max `window.innerWidth - 320` so the tab content stays readable. `clampChatPaneWidth()` re-clamps on read of persisted prefs in case the viewport has shrunk between sessions.

### Endpoints

Mirrors Math Symbols' `/api/synthesis/*`, `/api/training/*`, `/api/inference/render`, `/api/device/*`, plus:

- `GET /api/state` / `POST /api/state/patch` — read or deep-merge into the pipeline-state mirror.
- `WS /ws/state` — first frame is a `state_replace` snapshot; thereafter `state_patch` per mutation, plus `training_session` / `training_tick` / `validation_tick` for live agent-driven training progress.
- `POST /api/agent/chat` — run one ML Engineer turn (SSE stream of `text_delta` / `text_message` / `tool_use` / `tool_result` / `usage` / `result` / `error`).
- `POST /api/agent/stop` — cancel the in-flight turn.
- `GET /api/agent/sessions` and `/sessions/{id}/messages` — list past chat sessions and reconstruct transcripts (raw SDK messages normalized into the same event shape live streams use, so the frontend renderer has one code path).

### Backend file layout

`agent_state.py` (pipeline-state mirror + WS broadcast + `broadcast_event` helper for non-state events), `agent_runtime.py` (one turn through the SDK), `agent_tools.py` (MCP tool definitions + session/device accessor wiring), `agent_system_prompt.md` (the ML Engineer's persona and workflow patterns), plus the unchanged `ml.py`, `synthesis.py`, `training.py`, `server.py` from Math Symbols. No `training_worker.py` (training is synchronous, same as Math Symbols).

### Frontend file layout

Same six-tab structure as Math Symbols (`OrientationTab`, `DataSynthesisTab`, `ArchitectureTab`, `TrainingTab`, `InferenceTab`, `DebriefTab`). New components under `components/chat/`: `ChatPane`, `ChatTranscript`, `UsageBar`, `SessionMenu`, plus `toolLabels.ts` for human-readable tool names. State stores: `chat` (visible, items, activeSessionId, sessions, turn, usage, draft, paneWidth) added alongside the existing `synthesis` / `architecture` / `training` stores.

**Ports.** Backend on `5041`, Vite dev on `5042`. Override via `AGENTIC_SERVER_PORT`. Only one of the four GUI apps can run at a time without overriding ports.

**Extra Python deps.** `claude-agent-sdk` for the in-process MCP tool server and `query()` driver.

## Agentic Snake Trainer (`05-agentic-snake/`)

The curriculum's first **reinforcement-learning** project, and (like `04`) an
agentic one: an embedded **RL Coach** agent (Claude Opus 4.8, 1M context) drives
the same pipeline the user can. An RL agent learns to play **Snake** on a grid —
no dataset; experience is generated by playing. The agent layer (state mirror +
runtime + MCP tools + chat pane) is ported from `04-agentic-symbols`; the RL core
is new.

### The RL core (no GUI dependency)

- **`game.py`** — `SnakeEnv` on a configurable grid (default 10×10). Action space
  is **3 relative actions** (straight / turn-right / turn-left) so a 180° suicide
  is impossible and the policy is orientation-invariant. Reward is a tunable
  `RewardConfig` (food / death / per-step / toward-food / away-from-food shaping).
  `EnvConfig.observation` selects one of **two observation models** (the data
  the agent's policy is a function of), with `state_shape(cfg)` reporting the
  resulting shape:
  - `"features"` (default) — an engineered **11-feature vector** (danger ×3,
    heading ×4, food-direction ×4). Discrete and tiny so tabular Q-learning is
    feasible; its blind spot is it only probes the 3 cells next to the head, so
    the agent can't see its own body shape and traps itself. `STATE_SIZE=11`.
  - `"grid"` — the full board as a `(GRID_CHANNELS=3, H, W)` tensor (body / head
    / food). The agent sees its whole body; cost is a far larger state, so
    tabular Q-learning is rejected (the canonical motivation for function
    approximation) and the deep agents use a CNN and need more training.

  Both observations feed the same **3 relative actions** so DQN/REINFORCE stay
  comparable across models. `NUM_ACTIONS=3`. `render_dict()` is the JSON frame.
- **`models.py`** — `pick_device`, `CKPT_DIR`, and `QNetwork` / `PolicyNetwork`,
  which take an `obs_shape` and dispatch to an MLP (1-D features) or a small CNN
  (3-D grid) via `_build_net`; both end in a `NUM_ACTIONS` head so the agents are
  body-agnostic. (Feature-mode MLP keys are unchanged, so old checkpoints load.)
- **`agents.py`** — three agents behind one `act / observe / end_episode /
  action_scores` interface: `QLearningAgent` (tabular dict of Q-values, no NN),
  `DQNAgent` (replay buffer + target net + ε-decay), `ReinforceAgent` (Monte-Carlo
  policy gradient, learns at episode end). `action_scores(state)` returns
  `{kind:'q'|'prob', values}` for the Watch-tab overlay. `ALGORITHMS` registry +
  `build_agent` + `default_hyperparams` drive the catalog endpoint. `build_agent`
  takes `obs_shape`; `QLearningAgent` raises `ValueError` if handed a grid shape
  (surfaced as a 400 by the init endpoint / a tool error by the agent).
- **`train.py`** — headless CLI ground truth: `uv run python 05-agentic-snake/train.py
  --algo qlearning|dqn|reinforce --episodes N [--observation features|grid] [--device cpu]`.
  Reports rolling-mean score + greedy eval. (Validated: all three learn on the
  feature obs — Q-learning ~18 avg, DQN ~15, REINFORCE ~17 greedy apples on
  10×10; grid DQN learns more slowly — ~4 greedy at 500 eps, climbing — since the
  CNN over absolute coordinates is a harder problem that wants more episodes.)
- **`training.py`** — `SnakeSession` (env + agent + per-episode `score_history`),
  the unit both drivers build on. `train_one_episode()` (the atom),
  `train_episodes(n, on_episode)`, `evaluate(n)` (greedy), `play_episode()`
  (records every frame + per-step `action_scores` for the Watch tab),
  `update_hyperparameters` (hot-swaps lr), `move_to_device`, plus
  save/load/list/delete checkpoint helpers. Checkpoints persist algo +
  hyperparameters + env_config + agent state + score history.

### Backend (mirrors `04`, retargeted to RL)

- **Synchronous REST, no worker thread** (same as `02`/`04`). Training runs in
  bounded chunks; the human UI loops `POST /api/training/train_episodes` (≤100/call)
  with a Stop flag, the agent loops its `train_n_episodes` tool (≤200/call).
  *Deviation from the original plan*, which assumed the MNIST worker+WS pattern —
  the combined agentic app uses `04`'s synchronous model so the agent and UI share
  one code path.
- **`agent_state.py`** — pipeline-state mirror (`environment` / `algorithm` /
  `training` slices) + `/ws/state` broadcast, ported verbatim from `04`.
- **`server.py`** — REST + state mirror + agent SSE. Snake-specific endpoints:
  `GET /api/catalog` (algorithms + default hyperparameters), `POST
  /api/training/init` (builds a `SnakeSession` from the **state mirror**, not a
  request body), `train_episodes`, `eval`, `play` (frames + overlay), checkpoints,
  `device/*`. The REST init/reset/load endpoints **don't** broadcast
  `training_session` — the initiating tab updates itself from the response, and an
  empty-history echo could race its first pushed episodes; only the agent path
  broadcasts (so the UI reflects agent-driven changes it didn't initiate).
- **`agent_tools.py`** — 17 MCP tools for the RL Coach: read (`get_pipeline_state`,
  `get_recent_progress`, `list_algorithms/checkpoints/devices`), config
  (`set_environment`, `set_algorithm`, `set_hyperparameters`), training
  (`init_session`, `reset_session`, `train_n_episodes` — broadcasts an
  `episode_tick` per episode with `await asyncio.sleep(0)`, `evaluate`,
  `watch_agent_play` — returns score/length/how-it-ended for failure diagnosis),
  checkpoints, `select_device`. `agent_runtime.py` + `agent_system_prompt.md`
  (the RL Coach persona) ported from `04`; model pinned to `claude-opus-4-8[1m]`,
  MCP server name `agentic-snake`.

### Frontend (six tabs + chat pane)

Forked from `04`'s frontend — `Header`, `StatusBar`, `DeviceSelector`,
`ThemeSelector`, and the entire `components/chat/` folder (ChatPane / ChatTranscript
/ UsageBar / SessionMenu / toolLabels) are domain-agnostic and reused; `state.svelte.ts`,
`api.ts`, and `App.svelte` were rewritten for the RL domain. Stores: `environment`,
`algorithm` (+ `catalog`), `training` (RL session + `scoreHistory` + run-state
lifted to the store so the train loop survives tab remount), `chat`. Tabs:
**0 Orientation · 1 Environment** (grid + reward sliders + live preview board) **·
2 Algorithm** (3-card picker + per-algo hyperparameter editor) **· 3 Training**
(Initialize/Re-initialize Agent, Episodes input, Train N / Continuous-with-Stop,
greedy Evaluate, checkpoints sidebar, live `ScoreChart`s for score + survival
length) **· 4 Watch** (`GameBoard` canvas animating a `play_episode` result at
adjustable fps, with a per-step Q-value / action-probability overlay showing what
the agent "thinks") **· 5 Debrief** (status-bucket summary + setup/progress stats).
New shared components: `GameBoard.svelte` (canvas renderer) and `ScoreChart.svelte`
(SVG raw + rolling-mean line). The WS reducer in `App.svelte` handles
`state_replace`/`state_patch` (mirror) plus `training_session`/`episode_tick`
(live agent-driven progress).

**Ports.** Backend `5041`, Vite dev `5042`. Override `SNAKE_SERVER_PORT`.

**Extra deps.** None beyond the shared stack — `claude-agent-sdk` was already in
`pyproject.toml`; the frontend dropped `04`'s `katex` dependency.

## Agentic Cube Trainer (`06-agentic-cube/`)

The curriculum's **second RL project** and an agentic one: an embedded **RL Coach**
(Claude Opus 4.8, 1M context, MCP server name `agentic-cube`) drives a pipeline
that teaches a network to **solve a Rubik's Cube**. Forked from `05-agentic-snake`
— the agent layer (state mirror, runtime, chat pane, six-tab shell) is reused; the
RL core is new, and three subsystems are genuinely new (background run, check-in
scheduler, live report). The headline lesson is the **sparse-reward problem** and
its fix: a scrambled cube has one goal among ~4.3×10¹⁹ states, so model-free RL
never finds it — instead we exploit the cube's known model with **value iteration
over a learned cost-to-go function** (DeepCubeA-lite), trained on a **reverse-
scramble curriculum** and paired with beam search at solve time.

### The RL core (no GUI dependency)

- **`cube.py`** — `CubeEnv`-style facelet model for a `size` ∈ {2,3} cube. State is
  a flat `int8` array of `6·size²` facelet colors; moves are **precomputed
  permutations** (`state[perm]`) built once from an exact right-hand-rule cubie
  geometry, so they're provably a valid group (self-test in `__main__`:
  `move⁴=identity`, `move·move'=identity`, scrambles reverse exactly). 6 moves for
  2×2, 12 for 3×3. `scramble(k)`, `is_solved` (faces-uniform), `encode_batch`
  (one-hot), `render_dict` (per-cubie sticker data + `move_meta` axis/sign/dir for
  the 3D renderer).
- **`models.py`** — `pick_device`, `CKPT_DIR`, and `CostToGoNet` (an MLP over the
  one-hot facelets with a **single scalar** moves-to-solve output). No CNN branch
  (cube obs is a flat vector).
- **`agents.py`** — one `ValueIterationAgent`: `train_batch` expands every child,
  scores with a **target net**, regresses the online net to `0 if solved else
  1+min_child_cost` (MSE); `solve(state, beam_width)` is a heuristic-guided **beam
  search**; `action_scores` returns per-move cost-to-go for the Watch overlay.
  `ALGORITHMS = {"value_iteration": …}`, `build_agent`, `default_hyperparams`.
- **`training.py`** — `CubeSession`: unit of work is a **value-iteration
  iteration** (not an episode). `train_one_iteration(k)` / `train_iterations` /
  `evaluate(n,k)` (returns **solve-rate** + mean solution length) / `solve_episode`
  (records every cube state + per-step move + scores for the 3D Watch) /
  `update_hyperparameters` / checkpoint I/O (persists algo, hp, cube_config, agent
  state, `iteration`, `current_k`, `solve_rate_by_k`, `loss_history`).
- **`train.py`** — headless CLI ground truth: `uv run python 06-agentic-cube/train.py
  --cube 2|3 --iterations N [--start-k K --max-k K --promote-at 0.9]`. Ramps the
  curriculum on solve-rate. (Validated: 2×2 reaches **100% solve-rate at all depths
  ≤14** in ~1k iters / ~40s CPU; 3×3 learns shallow depths fast and pushes deeper
  over a long run.) Used to produce the bundled `checkpoints/pretrained-2x2.pt`
  (and best-effort `pretrained-3x3.pt`) the Watch tab loads by default.

### New backend subsystems (beyond the `05`/`04` shape)

- **`background_trainer.py`** — `BackgroundTrainer`, a process-owned daemon
  `threading.Thread` that survives chat turns / browser disconnects (overnight-
  capable). Advances the curriculum (train at `k` → eval → promote when solve-rate
  ≥ `promote_at`), auto-checkpoints (`autorun_<run_id>.pt` + a
  `reports/run_<id>.manifest.json` for resume), and is stoppable. A shared
  `threading.RLock` (`SESSION_LOCK`) guards every `CubeSession` access; the thread
  reaches the WS **only** via `agent_state.broadcast_event_threadsafe`
  (`loop.call_soon_threadsafe` — the subscriber queues are loop-affine). Raises
  milestones (k-promotion, checkpoint, finish) for the scheduler.
- **`checkin_scheduler.py`** — a server-side asyncio task that wakes the RL Coach on
  a **hybrid cadence** (milestone OR ≥ `cadence_minutes`), runs one short `run_turn`
  resuming the **same coach session**, and bridges its events onto `/ws/state` as
  `agent_event` (so check-ins fire headless, browser closed; an open chat renders
  them live). Shares `AGENT_TURN_LOCK` (in `agent_runtime.py`) with the user chat
  path so a user turn and an autonomous check-in never interleave (user wins).
- **`report_store.py`** — the live markdown training report the Coach maintains
  (`update_training_report`) plus a finalized debrief (`generate_final_report`);
  broadcasts `report_update` / `report_final`; `GET /api/training/report` for
  hydration. Replaces `02`–`05`'s pre-packaged status-bucket debrief.

### Backend (mirrors `05`, retargeted)

- `agent_state.py` slices: `environment` (`size` + `curriculum:{startK,maxK,
  promoteAt}`), `algorithm`, `training` (`iterationsPerRun, evalEveryN, evalN,
  cadenceMinutes`). Adds `set_loop` + `broadcast_event_threadsafe`.
- `server.py` REST: the usual `/api/training/{init,train_iterations,eval,play,
  reset,checkpoints/*}` + **run control** `/api/training/run/{start,stop,status,
  resume}` + `GET /api/training/report`; a `startup` hook captures the loop and
  launches the scheduler; the SSE chat handler holds `AGENT_TURN_LOCK` and records
  the coach `session_id` for autonomous check-ins. **`reload=True` would kill the
  background thread on file save — set `CUBE_NO_RELOAD=1` for overnight runs.**
- `agent_tools.py` — 24 MCP tools: reads (incl. `get_run_status`), config
  (`set_environment` cube+curriculum, `set_hyperparameters`), foreground
  (`init_session`, `train_n_iterations`, `set_curriculum_depth` (jump the live
  session's `current_k`), `evaluate`, `watch_agent_play`), **run control**
  (`start_training_run`, `stop_training_run`, `set_curriculum_schedule`),
  **report** (`update_training_report`, `generate_final_report`), checkpoints,
  device. Session-touching tools take `SESSION_LOCK`; foreground training refuses
  while a background run owns the session.
- **Curriculum `k` is user/agent-settable.** `current_k` advances automatically
  during a background run (promote on solve-rate ≥ `promoteAt`); the Training tab
  also exposes a manual depth stepper + an "auto-advance when solved" toggle for
  foreground training (`train_iterations` persists a passed `k` onto the session),
  and the agent can jump it via `set_curriculum_depth`.
- **Bundled checkpoints are protected.** `training.PROTECTED_CHECKPOINTS`
  (`pretrained-2x2.pt`, `pretrained-3x3.pt`, committed and loaded by the Watch tab
  by default) can't be deleted — `delete_checkpoint` raises, `list_checkpoints`
  marks them `protected`, the UI shows a lock instead of an ✕, and other deletes
  go through a confirm dialog.

### Frontend (seven tabs + chat pane)

Reuses `Header` (logo is a custom isometric 3×3×3 `cube` icon in `Icon.svelte`,
two stroke weights), `StatusBar`, `DeviceSelector`, `ThemeSelector`, `ScoreChart`,
and the whole `components/chat/` folder (the chat event reducer is extracted to
`chat/chatReducer.ts` so the SSE path AND the WS-bridged `agent_event` path share
one code path). Stores: `cube`, `algorithm`, `training` (RL session +
`lossHistory` + `solveRateByK` + foreground run-state + background `run` snapshot
+ `autoAdvance`), `report`, `chat`, plus the persisted `cubeStyle` and `sound`
prefs.

Tabs: **0 Orientation · 1 Cube** (size + curriculum sliders + live 3D preview) **·
2 Algorithm · 3 Training · 4 Watch · 5 Progress Report** (live report via shared
`Markdown.svelte` + **Finish Training** → drives `generate_final_report` →
navigates to Debrief) **· 6 Debrief** (Coach-written final report + fact cards).
(Tab labels align across the row regardless of subtitle via a `min-height` on
`.tab-subtitle`.)

- **Training tab** has **Foreground / Background sub-tabs**. Foreground: a manual
  scramble-depth **k stepper** + an **"auto-advance k when solved"** toggle (which
  evaluates every `evalEveryN` and promotes at `promoteAt`, mirroring the
  background run's rule), Train N / Continuous / Stop. Background: cadence + Start/
  Stop Overnight Run + live run status. Right column: stat cards, a loss
  `ScoreChart`, and a 2-up row of **solve-rate-by-k** + a **"Play last training
  round"** 3D view (the current model solving a depth-`k` scramble). The standalone
  "Evaluate" button was removed (solve-rate now comes from auto-advance / the
  background run).

3D rendering:
- **`CubeView3D.svelte`** — a real **three.js** cube built from per-cubie meshes,
  `animateMove` physically rotating the affected layer 90° (pivot + tween + bake),
  `OrbitControls` for drag/zoom. Reads the selected **cube style** and rebuilds
  materials live on change (deduped material disposal).
- **`CubePlayer.svelte`** — shared 3D viewer + transport (play/pause/restart, fps,
  moves slider) over a `SolveResult`; used by both the Watch tab and the Training
  tab's "Play last training round". `stepIndex` is bindable so the Watch tab's
  per-step cost-to-go overlay can read it.
- `cubeGeometry.ts` (`solvedFrame` for static previews), `Markdown.svelte`.

**Cube styles** (`cubeStyles.ts` + `CubeStyleSelector.svelte`): six face-color
schemes — **Standard / Jewel / Neon** and a semi-transparent "**(Glass)**" variant
of each (≈70% opacity, plastic body dropped, `depthWrite` off so you see through).
Selected in the status bar, persisted (`agentic-cube.cube-style.v1`).

**Sound** (`sound.ts` + `SoundToggle.svelte`): synthesized Web Audio effects (no
assets) — a **solve chime** (ascending arpeggio), a per-turn **clack**, and an
**unsolved whiff** — fired from `CubePlayer`'s play loop (so both Watch and
Training get them); `resumeAudio()` runs inside the Play click for autoplay
policy. A master on/off toggle in the status bar, persisted
(`agentic-cube.sound.v1`).

The **status bar** shows live run state on the left and the **Sound · Cube ·
Theme** controls bottom-right. The WS reducer in `App.svelte` handles `state_*` +
`training_session` + `trainer_progress` + `trainer_status` + `report_update`/
`report_final` + `agent_event`.

**Ports.** Backend `5041`, Vite dev `5042`. Override `CUBE_SERVER_PORT`.

**Extra deps.** Frontend adds **`three`** (three.js + `OrbitControls`) for the 3D
views; dropped `katex`. No new Python deps (`claude-agent-sdk` already present).

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

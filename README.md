# ML Learning Apps

A growing suite of small, self-contained applications for learning and experimenting with machine learning. Each project trains a real neural network end-to-end and pairs the training loop with an interactive web UI so the underlying ideas — data, architecture, optimization, generalization — are visible and tunable in real time.

The goal is pedagogical, not industrial: features earn their keep by making something about ML easier to *see* or *feel* (live loss curves, hyperparameters as sliders, freehand inference, data inspection). Projects are numbered in curriculum order and progress from the basics toward what the local hardware can handle.

## Apps

### `01-mnist-trainer/` — MNIST digit classifier

The starting point. Train a small MLP or CNN on MNIST while watching loss and validation curves update live. Features a draw-to-classify pad, a virtualized data explorer for browsing the training set, model checkpoint save/load, and a CLI (`train.py`) that mirrors the GUI.

### `02-math-symbols/` — Math symbol OCR

A printed-symbol classifier covering the glyphs used in mathematical writing — digits, alphanumerics, punctuation, Greek upper/lower, and math symbols (potentially several hundred classes). Distinct from MNIST in three ways:

- **Synthesized data, not downloaded.** Glyphs are rendered on the fly from system/open-source fonts during training. No fixed dataset on disk. Augmentation (noise, skew) is a live, tunable hyperparameter.
- **Held-out fonts for validation.** Generalization is measured against fonts the training pipeline never sees, so val accuracy reflects real generalization rather than memorization of font-class pairs.
- **Pedagogical tabbed UI** that walks through the full pipeline:
  - **Orientation** — overview and entry point.
  - **Data Synthesis** — pick symbol categories, training fonts, validation fonts, and augmentation; preview the synthesized images live.
  - **Model Architecture** — design a CNN layer-by-layer with drag-and-drop, or click *Suggest* for an architecture matched to the current data config.
  - **Training** — initialize the model, run single batches (with an animated per-image classification sweep), full epochs, or train continuously while watching training and validation loss curves stream in. Save / load checkpoints (including auto-save and auto-load on restart).
  - **Inference** — type free-form text with `$…$` math delimiters; KaTeX renders the preview, the model classifies each glyph, and the UI shows top-K predictions per cell with rendered alternatives and confidences.
  - **Debrief** — a contextual summary that describes the user's current pipeline state and tells them what to do next, with messages that adapt as training accuracy improves.

Checkpoints save the entire training pipeline (synthesis config + architecture + weights + loss histories) so you can suspend and resume across sessions without losing context.

## Stack

- **PyTorch** on Apple Silicon (MPS); CUDA / CPU paths are kept as fallbacks. Python 3.13.
- **uv** for Python dependency management — one shared virtual env at the repo root, since PyTorch is large and reinstalling per project is painful.
- **FastAPI** backends serving REST (and, in the MNIST app, a WebSocket for live training events).
- **Svelte 5 + Vite + Tailwind** frontends, served by FastAPI as static `dist/` in production and proxied through the Vite dev server in development.

## Quick start

Clone, install Python deps, run an app:

```bash
git clone https://github.com/danromik/machine-learning-apps.git
cd machine-learning-apps
uv sync                                    # creates .venv/ at repo root

# run an app — builds frontend, starts backend, opens browser
01-mnist-trainer/run.sh
# or
02-math-symbols/run.sh
```

Both apps default to `http://localhost:5041`. Only one can run at a time without overriding `MNIST_SERVER_PORT` / `MATH_SERVER_PORT`.

For headless / scriptable training, the MNIST project also has a CLI:

```bash
uv run python 01-mnist-trainer/train.py --model cnn --epochs 5
```

## Repo layout

```
.
├── pyproject.toml                # shared Python deps
├── 01-mnist-trainer/             # MLP / CNN trained on MNIST
├── 02-math-symbols/              # OCR over synthesized glyphs
└── NN-<next-project>/            # future projects, numbered in curriculum order
```

Each project is fully self-contained — its own `data/`, `checkpoints/`, models, training loop. Projects can be deleted, copied, or forked without affecting siblings.

## License

[MIT](LICENSE).

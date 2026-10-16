# 07 — Language Model (TinyStories GPT)

A small GPT-style **decoder-only transformer** trained from scratch on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). That corpus
holds about 2.7M short children's stories written by GPT-3.5/GPT-4 using a small
vocabulary. It's narrow enough that a ~14M-parameter model learns to write
coherent little stories on a laptop GPU in a few hours.

This project is **CLI-only by design**. It has no web UI. The teaching surface is a
live terminal dashboard (built with `rich`) that shows training progress, the loss
sparkline, and text samples that improve as training goes on.

## The idea

Earlier projects classify an input or choose an action. This one learns
**next-token prediction**: given a sequence of tokens, predict a probability
distribution over the next one. The pieces are:

1. **Tokenization.** A byte-level **BPE** tokenizer (8192 tokens, trained with
   HuggingFace `tokenizers` on the validation split) turns text into integer IDs.
   Stories are separated by an `<|endoftext|>` token.
2. **Model.** A GPT-2-style transformer (`model.py`) with learned positional
   embeddings and pre-LayerNorm blocks. Each block has causal multi-head
   self-attention (`F.scaled_dot_product_attention`, `is_causal=True`) and a GELU
   MLP. The input embedding and output projection share weights, and the residual
   projections use scaled init. Default size: 6 layers × 6 heads × 384 dims, a
   256-token context, and ~14M parameters.
3. **Training.** Cross-entropy on shifted sequences (each position predicts the
   next token), with AdamW (weight decay only on ≥2-D weights), linear warmup then
   cosine decay to 10% of peak LR, and gradient clipping at 1.0. Batches are random
   256-token windows sampled from a memory-mapped token file.
4. **Sampling.** Autoregressive generation with temperature + top-k, stopping at
   `<|endoftext|>`.

A sanity check worth knowing: an untrained model should start at loss ≈ ln(8192) ≈ 9.01,
which means it is uniform over the vocabulary. `python model.py` prints both numbers.

## Quick start

All commands run from the repo root with the shared env.

```bash
# 1. Download + tokenize the corpus (train is ~2.2 GB raw → ~1.1 GB of uint16 tokens)
uv run python 07-languagemodel/data.py valid
uv run python 07-languagemodel/data.py train

# 2. Train (default: 5000 steps; opens the live dashboard)
uv run python 07-languagemodel/train.py --name mytest

# 3. Generate stories from a checkpoint
uv run python 07-languagemodel/sample.py 07-languagemodel/runs/<run>/ckpt-best.pt \
    --prompt "Once upon a time" --num-samples 3 --temperature 0.8
```

The trained tokenizer (`tokenizer/bpe-8192.json`) is committed, so step 1 works on
a fresh clone. To retrain it, run `uv run python 07-languagemodel/tokenizer.py` (it
needs the valid split downloaded and skips training if the file already exists).
Re-encoding after a tokenizer change requires deleting the `.bin` files in `data/`.

### Training CLI

```
--name NAME            run name suffix (run dir: runs/<timestamp>-<name>/)
--resume PATH          resume from a checkpoint (model, optimizer, RNG, history)
--device mps|cuda|cpu  override auto-detection
--n-layer / --n-head / --n-embd / --block-size / --dropout / --vocab-size
--batch-size / --lr / --max-steps / --warmup-steps
--eval-every / --sample-every / --checkpoint-every
```

`--resume` restores the saved model and training configs, but it takes
`--max-steps` from the command line. That way a finished run can be **extended**:
the cosine schedule stretches to the new length.

### Live controls (while training)

| key | action |
|---|---|
| `space` | pause / resume |
| `s` | save `ckpt-last.pt` now |
| `g` | generate a sample now |
| `e` | run a validation eval now |
| `+` / `-` | multiply / divide the peak LR by 1.5 (hot-swap) |
| `q` | save and quit |

## Outputs

Each run writes to `runs/<timestamp>[-name]/`:

- `config.json`: the model + training config, device, and resume source.
- `metrics.jsonl`: one record per eval (`step, train_loss, val_loss, lr, elapsed`).
- `samples.txt`: the "Once upon a time…" sample at every `sample_every` step, so
  you can read how the model's writing improves over training.
- `ckpt-best.pt`: saved whenever the val loss improves. `ckpt-last.pt` is saved
  periodically and at exit. Checkpoints are resumable: they hold the weights,
  optimizer state, configs, step, elapsed time, best val, history, and RNG states.
  Saves are atomic (written to a temp file, then renamed).

`data/` and `runs/` are gitignored.

## Reference run

One full pass over the training set (65k steps × 32 × 256 ≈ 532M tokens, warmup
1000) on an Apple Silicon Mac (MPS):

```bash
uv run python 07-languagemodel/train.py --name fullpass \
    --max-steps 65000 --warmup-steps 1000 \
    --eval-every 1000 --sample-every 1000 --checkpoint-every 2000
```

- ~4.8 hours. Val loss fell from 2.47 at step 1k to **1.37** at the end
  (train loss ~1.41).
- At step 65k, the sample for the prompt "Once upon a time" came out grammatical and on topic:

  > Once upon a time, there was a little girl named Lily. She loved to play outside
  > with her friends. One day, she saw a big, round melon in the garden. Lily was
  > worried because she didn't know where the melon came from.

## Files

| file | role |
|---|---|
| `paths.py` | single source of truth for `DATA_DIR`, `TOKENIZER_DIR`, `CKPT_DIR`, `RUNS_DIR` (created on import) |
| `tokenizer.py` | train / load the byte-level BPE tokenizer |
| `data.py` | download TinyStories, encode a split to a flat `uint16` `.bin`, memmap loader, `get_batch` |
| `model.py` | `GPTConfig`, `GPT` (attention / MLP / block), `generate`, `configure_optimizers`, `pick_device` |
| `engine.py` | `TrainConfig`, the LR schedule (`lr_at`), `train_step`, `estimate_loss` |
| `checkpoint.py` | atomic save, load, `rebuild` (model + optimizer + configs + RNG) |
| `console.py` | `rich` dashboard, sparkline, non-blocking `KeyReader` |
| `train.py` | the training CLI (run dirs, eval/sample/checkpoint cadence, key controls) |
| `sample.py` | generate text from a checkpoint |

Most modules also run on their own as smoke tests: `python model.py` (one forward
pass + expected initial loss), `python engine.py` (200-step mini-train + sample),
and `python tokenizer.py` (encode/decode round trip).

<script lang="ts">
  import { dataset, isDatasetReady } from '../../state.svelte';
  import { api, streamDownload, streamSamples, type Sample } from '../../api';

  let downloadError = $state<string | null>(null);
  let previewModalOpen = $state(false);
  let previewSplit = $state<'train' | 'val'>('train');
  let previewSamples = $state<Sample[]>([]);
  let previewBusy = $state(false);

  let ready = $derived(isDatasetReady());

  // ── Download ─────────────────────────────────────────────────────────

  async function startDownload() {
    if (dataset.downloading) return;
    downloadError = null;
    dataset.downloading = true;
    dataset.downloadStage = 'download';
    dataset.downloadFraction = 0;
    dataset.downloadError = null;
    try {
      for await (const ev of streamDownload()) {
        if ('error' in ev) {
          dataset.downloadError = ev.error;
          downloadError = ev.error;
          continue;
        }
        if ('done' in ev) {
          dataset.status = ev.status;
          continue;
        }
        dataset.downloadStage = ev.stage;
        dataset.downloadFraction = ev.fraction;
      }
      // Final status fetch — extract events don't always land if extraction
      // is fast.
      dataset.status = await api.datasetStatus();
    } catch (e) {
      downloadError = (e as Error).message;
    } finally {
      dataset.downloading = false;
      dataset.downloadStage = null;
    }
  }

  function fmtBytes(n: number): string {
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
  }

  // ── Preview modal ────────────────────────────────────────────────────

  async function openPreview(split: 'train' | 'val') {
    if (!ready) return;
    previewSplit = split;
    previewModalOpen = true;
    previewSamples = [];
    previewBusy = true;
    try {
      for await (const s of streamSamples({
        split,
        count: 50,
        seed: Math.floor(Math.random() * 1e6),
        flip: dataset.augmentation.flip,
        jitter: dataset.augmentation.jitter,
        random_crop: dataset.augmentation.random_crop,
      })) {
        previewSamples = [...previewSamples, s];
      }
    } catch (e) {
      console.error('preview failed', e);
    } finally {
      previewBusy = false;
    }
  }

  function closePreview() {
    previewModalOpen = false;
  }
</script>

<div class="h-full overflow-auto">
  <div class="max-w-3xl mx-auto px-4 py-3 flex flex-col gap-3">
    <!-- Header / explainer -->
    <section class="card p-4 flex flex-col gap-2">
      <h2 class="text-base font-semibold text-[var(--color-heading)]">
        Imagenette dataset
      </h2>
      <p class="text-sm text-[var(--color-text)] leading-relaxed">
        Imagenette is fast.ai's 10-class subset of ImageNet — the same kind of
        natural images AlexNet was trained on, but small enough to download in
        seconds and train at interactive speed on a single GPU. It's the
        standard teaching stand-in for full ILSVRC.
      </p>
      <p class="text-xs text-[var(--color-muted)] leading-relaxed">
        Download size: ~88 MB. Source: <span class="font-mono">{dataset.status?.url ?? ''}</span>.
      </p>
    </section>

    <!-- Download / status -->
    <section class="card p-4 flex flex-col gap-3">
      <header class="flex items-baseline justify-between gap-3 flex-wrap">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Status
        </h3>
        {#if dataset.status}
          <span class="text-xs text-[var(--color-muted)] font-mono">
            {dataset.status.data_dir}
          </span>
        {/if}
      </header>

      {#if !dataset.loaded}
        <p class="text-sm text-[var(--color-muted)]">Loading…</p>
      {:else if !ready}
        <div class="flex flex-col gap-3">
          <p class="text-sm text-[var(--color-text)]">
            {#if dataset.status?.archive_present}
              Archive present
              ({fmtBytes(dataset.status.archive_size)}) —
              waiting on extraction.
            {:else}
              Dataset has not been downloaded yet.
            {/if}
          </p>
          {#if dataset.downloading}
            <div class="flex flex-col gap-1">
              <div class="flex justify-between text-xs text-[var(--color-muted)] font-mono">
                <span>{dataset.downloadStage ?? '…'}</span>
                <span>{Math.round(dataset.downloadFraction * 100)}%</span>
              </div>
              <div class="h-2 rounded bg-[var(--color-surface-2)] overflow-hidden">
                <div
                  class="h-full bg-[var(--color-accent)] transition-[width]"
                  style="width: {Math.round(dataset.downloadFraction * 100)}%;"
                ></div>
              </div>
            </div>
          {:else}
            <button
              type="button"
              class="btn-primary self-start"
              onclick={startDownload}
            >
              Download Imagenette
            </button>
          {/if}
          {#if downloadError}
            <p class="text-xs text-[var(--color-danger)]">{downloadError}</p>
          {/if}
        </div>
      {:else}
        <div class="flex flex-col gap-2">
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Train images</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {dataset.status?.num_train.toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">Val images</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {dataset.status?.num_val.toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">On disk</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {fmtBytes(dataset.status?.extract_size ?? 0)}
            </dd>
          </dl>
        </div>
      {/if}
    </section>

    <!-- Class table -->
    <section class="card p-4 flex flex-col gap-2">
      <h3 class="text-sm font-semibold text-[var(--color-heading)]">
        Classes ({dataset.classes.length})
      </h3>
      <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-sm">
        {#each dataset.classes as c, i (c.wnid)}
          {@const counts = dataset.status?.per_class?.[i]}
          <div class="flex items-baseline gap-2 min-w-0">
            <span
              class="font-mono text-xs text-[var(--color-muted)] tabular-nums w-4 shrink-0"
              >{i}</span>
            <span class="text-[var(--color-text)] truncate">{c.label}</span>
            {#if counts && counts.train > 0}
              <span class="text-[10px] text-[var(--color-muted)] font-mono ml-auto shrink-0">
                {counts.train}/{counts.val}
              </span>
            {/if}
          </div>
        {/each}
      </div>
    </section>

    <!-- Augmentation controls -->
    <section class="card p-4 flex flex-col gap-3">
      <h3 class="text-sm font-semibold text-[var(--color-heading)]">
        Augmentation (training pipeline only)
      </h3>
      <p class="text-xs text-[var(--color-muted)] leading-relaxed">
        These transforms are applied to images on the fly during training, to
        regularize the model. Validation images go through deterministic
        center-crop only.
      </p>
      <div class="flex flex-col gap-2">
        <label class="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={dataset.augmentation.random_crop}
            onchange={(e) =>
              (dataset.augmentation.random_crop = (e.currentTarget as HTMLInputElement).checked)}
          />
          <span>Random crop (resize shorter side to {Math.round(dataset.inputSize * 1.15)} px → crop {dataset.inputSize}×{dataset.inputSize})</span>
        </label>
        <label class="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={dataset.augmentation.flip}
            onchange={(e) =>
              (dataset.augmentation.flip = (e.currentTarget as HTMLInputElement).checked)}
          />
          <span>Random horizontal flip (50%)</span>
        </label>
        <label class="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={dataset.augmentation.jitter > 0}
            onchange={(e) =>
              (dataset.augmentation.jitter = (e.currentTarget as HTMLInputElement).checked
                ? 0.4
                : 0)}
          />
          <span>Color jitter (brightness · contrast · saturation)</span>
        </label>
        {#if dataset.augmentation.jitter > 0}
          <label class="flex items-center gap-2 text-sm pl-6">
            <span class="text-[var(--color-muted)] text-xs w-16">strength</span>
            <input
              type="range"
              min="0.05"
              max="1.0"
              step="0.05"
              value={dataset.augmentation.jitter}
              oninput={(e) =>
                (dataset.augmentation.jitter = parseFloat(
                  (e.currentTarget as HTMLInputElement).value
                ))}
              class="flex-1"
            />
            <span class="text-xs font-mono w-10 text-right">
              {dataset.augmentation.jitter.toFixed(2)}
            </span>
          </label>
        {/if}
      </div>
    </section>

    <!-- Preview -->
    {#if ready}
      <section class="card p-4 flex flex-col gap-2">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Preview
        </h3>
        <p class="text-xs text-[var(--color-muted)]">
          Stream a sample of images through the training or validation
          pipeline to see exactly what the model will train against.
        </p>
        <div class="flex gap-2">
          <button
            type="button"
            class="btn-outline text-sm"
            onclick={() => openPreview('train')}
          >
            Preview Train
          </button>
          <button
            type="button"
            class="btn-outline text-sm"
            onclick={() => openPreview('val')}
          >
            Preview Val
          </button>
        </div>
      </section>
    {/if}
  </div>
</div>

{#if previewModalOpen}
  <div class="modal-backdrop" onclick={closePreview} role="presentation">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_interactive_supports_focus -->
    <div
      class="modal-panel"
      onclick={(e) => e.stopPropagation()}
      role="dialog"
      tabindex="-1"
      aria-label="Preview {previewSplit} images"
    >
      <header class="flex items-baseline justify-between gap-3 px-4 py-2 border-b border-[var(--color-border)]">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          {previewSplit === 'train' ? 'Train' : 'Validation'} preview
          <span class="text-[var(--color-muted)] font-normal">
            · {previewSamples.length} samples{previewBusy ? '…' : ''}
          </span>
        </h3>
        <button
          type="button"
          class="text-xs text-[var(--color-muted)] hover:text-[var(--color-text)]"
          onclick={closePreview}
        >
          ✕ Close
        </button>
      </header>
      <div class="flex-1 min-h-0 overflow-auto p-3">
        <div class="preview-grid">
          {#each previewSamples as s, i (i)}
            <div class="preview-cell" title="{s.label} · {s.source}">
              <img
                src="data:image/png;base64,{s.png_b64}"
                alt={s.label}
                class="preview-img"
              />
              <span class="preview-cell-label">{s.label}</span>
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
    padding: 1rem;
  }
  .modal-panel {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 0.5rem;
    width: 100%;
    max-width: 56rem;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  }
  .preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
    gap: 0.5rem;
  }
  .preview-cell {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
    align-items: center;
    padding: 0.25rem;
    border: 1px solid var(--color-border);
    border-radius: 0.25rem;
    background: var(--color-surface);
  }
  .preview-img {
    width: 96px;
    height: 96px;
    object-fit: cover;
    image-rendering: crisp-edges;
  }
  .preview-cell-label {
    font-size: 0.65rem;
    color: var(--color-muted);
    text-align: center;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>

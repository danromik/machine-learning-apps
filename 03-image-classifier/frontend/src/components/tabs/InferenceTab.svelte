<script lang="ts">
  import { api, type InferenceItem } from '../../api';
  import { isDatasetReady, training } from '../../state.svelte';

  let item = $state<InferenceItem | null>(null);
  let busy = $state(false);
  let errorMsg = $state<string | null>(null);
  let dropping = $state(false);

  let datasetReady = $derived(isDatasetReady());

  async function classifyFile(file: File) {
    if (!file.type.startsWith('image/')) {
      errorMsg = 'not an image file';
      return;
    }
    busy = true;
    errorMsg = null;
    try {
      item = await api.inferenceUpload(file);
    } catch (e) {
      errorMsg = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  function onFilePicked(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (file) classifyFile(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    dropping = false;
    const file = e.dataTransfer?.files?.[0];
    if (file) classifyFile(file);
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    dropping = true;
  }

  function onDragLeave() {
    dropping = false;
  }

  async function sampleVal() {
    busy = true;
    errorMsg = null;
    try {
      item = await api.inferenceSample();
    } catch (e) {
      errorMsg = (e as Error).message;
    } finally {
      busy = false;
    }
  }
</script>

<div class="h-full overflow-auto">
  <div class="max-w-4xl mx-auto px-4 py-3 flex flex-col gap-3">
    <!-- Source selector -->
    <section class="card p-4 flex flex-col gap-3">
      <h2 class="text-sm font-semibold text-[var(--color-heading)]">
        Classify an image
      </h2>
      {#if !training.hasSession}
        <p class="text-xs text-[var(--color-muted)]">
          No training session — initialize a model in the Training tab to get predictions.
          You can still preview how the input gets preprocessed.
        </p>
      {/if}
      <div class="flex flex-col sm:flex-row gap-3">
        <!-- Drop zone -->
        <label
          class="dropzone flex-1"
          class:dropzone-active={dropping}
          ondragover={onDragOver}
          ondragleave={onDragLeave}
          ondrop={onDrop}
        >
          <input
            type="file"
            accept="image/*"
            class="sr-only"
            onchange={onFilePicked}
          />
          <div class="text-sm text-[var(--color-text)] font-medium">
            Drop an image here
          </div>
          <div class="text-xs text-[var(--color-muted)]">
            or click to choose · resized + center-cropped to 96×96
          </div>
        </label>
        <!-- Sample-from-val button -->
        <button
          type="button"
          class="btn-outline"
          onclick={sampleVal}
          disabled={busy || !datasetReady}
          title={datasetReady
            ? 'Pick a random validation image and run inference on it'
            : 'Download the dataset first'}
        >
          Sample a val image →
        </button>
      </div>
      {#if errorMsg}
        <p class="text-xs text-[var(--color-danger)]">{errorMsg}</p>
      {/if}
    </section>

    {#if item}
      <section class="card p-4 flex flex-col gap-4">
        <div class="grid grid-cols-[auto_1fr] gap-6">
          <!-- Input image -->
          <div class="flex flex-col items-center gap-2">
            <span class="text-[10px] text-[var(--color-muted)] uppercase tracking-wide">
              input (model sees this)
            </span>
            <img
              src="data:image/png;base64,{item.input_png_b64}"
              alt="Input"
              class="block rounded border border-[var(--color-border)]"
              style="image-rendering: crisp-edges; width: 192px; height: 192px;"
            />
            {#if item.true_label}
              <span class="text-xs text-[var(--color-muted)]">
                true label: <span class="text-[var(--color-text)] font-medium">{item.true_label}</span>
              </span>
            {/if}
            {#if item.source}
              <span class="text-[10px] text-[var(--color-muted)] font-mono truncate max-w-[12rem]">
                {item.source}
              </span>
            {/if}
          </div>

          <!-- Predictions -->
          <div class="flex flex-col gap-3 min-w-0">
            {#if !item.has_session}
              <p class="text-sm text-[var(--color-muted)]">
                No training session — initialize a model to see predictions.
              </p>
            {:else if item.top_k.length === 0}
              <p class="text-sm text-[var(--color-muted)]">No predictions returned.</p>
            {:else}
              <div class="flex items-baseline justify-between">
                <h3 class="text-sm font-semibold text-[var(--color-heading)]">
                  Top {item.top_k.length} predictions
                </h3>
                {#if item.true_label && item.predicted_label}
                  {@const correct = item.predicted_label === item.true_label}
                  <span
                    class="text-xs font-medium"
                    style="color: {correct ? 'var(--color-success)' : 'var(--color-danger)'};"
                  >
                    {correct ? '✓ correct' : '✗ incorrect'}
                  </span>
                {/if}
              </div>
              <div class="flex flex-col gap-1.5">
                {#each item.top_k as alt, k (k)}
                  {@const isCorrect = item.true_label === alt.label}
                  {@const isTop = k === 0}
                  <div class="prediction-row">
                    <span class="prediction-rank">{k + 1}</span>
                    <span
                      class="prediction-label"
                      class:prediction-label-correct={isCorrect}
                    >
                      {alt.label}
                    </span>
                    <div class="prediction-bar">
                      <div
                        class="prediction-bar-fill"
                        style="
                          width: {(alt.confidence * 100).toFixed(1)}%;
                          background: {isCorrect
                            ? 'var(--color-success)'
                            : isTop && item.true_label && item.predicted_label !== item.true_label
                            ? 'var(--color-danger)'
                            : 'var(--color-accent)'};
                        "
                      ></div>
                    </div>
                    <span class="prediction-pct">
                      {(alt.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      </section>
    {:else if busy}
      <section class="card p-6 text-center text-sm text-[var(--color-muted)]">
        Classifying…
      </section>
    {/if}
  </div>
</div>

<style>
  .dropzone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
    padding: 1.5rem 1rem;
    border: 2px dashed var(--color-border);
    border-radius: 0.5rem;
    background: var(--color-surface);
    cursor: pointer;
    transition: background 80ms, border-color 80ms;
    text-align: center;
  }
  .dropzone:hover {
    background: var(--color-surface-2);
  }
  .dropzone-active {
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 8%, var(--color-surface));
  }
  .prediction-row {
    display: grid;
    grid-template-columns: 1.25rem 7rem 1fr 3rem;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
  }
  .prediction-rank {
    color: var(--color-muted);
    font-family: var(--font-mono);
    font-size: 0.625rem;
    text-align: right;
  }
  .prediction-label {
    color: var(--color-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .prediction-label-correct {
    color: var(--color-success);
    font-weight: 600;
  }
  .prediction-bar {
    height: 8px;
    background: var(--color-surface-2);
    border-radius: 4px;
    overflow: hidden;
  }
  .prediction-bar-fill {
    height: 100%;
    transition: width 120ms ease-out;
  }
  .prediction-pct {
    color: var(--color-muted);
    font-family: var(--font-mono);
    font-size: 0.625rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
</style>

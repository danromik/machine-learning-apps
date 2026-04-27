<script lang="ts">
  import { training } from '../../../state.svelte';

  let {
    onReinitialize,
    onTrainBatchFun,
    onTrainBatchFast,
    onTrainEpoch,
    onStopEpoch,
    onTrainContinuously,
    onStopContinuous,
    onSaveCheckpoint,
    onLoadCheckpoint,
    canTrain,
    busy,
    canReinit,
    reinitBlockedReason,
    batchesPerEpoch,
    epochRunning,
    continuousRunning,
  }: {
    onReinitialize: () => void;
    onTrainBatchFun: () => void;
    onTrainBatchFast: () => void;
    onTrainEpoch: () => void;
    onStopEpoch: () => void;
    onTrainContinuously: () => void;
    onStopContinuous: () => void;
    onSaveCheckpoint: () => void;
    onLoadCheckpoint: () => void;
    canTrain: boolean;
    busy: boolean;
    canReinit: boolean;
    reinitBlockedReason: string;
    batchesPerEpoch: number;
    epochRunning: boolean;
    continuousRunning: boolean;
  } = $props();

  let canLoad = $derived.by(() => {
    const f = training.checkpointFilename.trim();
    if (!f) return false;
    const withExt = f.endsWith('.pt') ? f : `${f}.pt`;
    return training.availableCheckpoints.includes(withExt);
  });
</script>

<aside
  class="w-48 shrink-0 flex flex-col gap-2 p-3 overflow-auto"
>
  <h3 class="text-sm font-semibold text-[var(--color-heading)]">
    Training Controls
  </h3>

  <div class="flex justify-center">
    <button
      type="button"
      class="btn-capsule"
      onclick={onReinitialize}
      disabled={busy || !canReinit}
      title={canReinit
        ? 'Build a fresh model from the current architecture and hyperparameters'
        : reinitBlockedReason}
    >
      Re-Initialize Model
    </button>
  </div>
  {#if !canReinit}
    <div
      class="text-[10px] text-[var(--color-danger)] leading-snug px-1 -mt-1 text-center"
    >
      {reinitBlockedReason}
    </div>
  {/if}

  <button
    type="button"
    class="btn-outline w-full text-xs"
    onclick={onTrainBatchFun}
    disabled={busy || !canTrain}
    title={canTrain
      ? 'Train one batch with the per-image highlight + prediction sweep'
      : 'Initialize the model first'}
  >
    Train 1 Batch (Fun)
  </button>

  <button
    type="button"
    class="btn-outline w-full text-xs"
    onclick={onTrainBatchFast}
    disabled={busy || !canTrain}
    title={canTrain
      ? 'Train one batch immediately, without the per-image animation'
      : 'Initialize the model first'}
  >
    Train 1 Batch (Fast)
  </button>

  {#if epochRunning}
    <button
      type="button"
      class="btn-danger w-full text-xs"
      onclick={onStopEpoch}
      title="Stop the current epoch at the next batch boundary"
    >
      Stop epoch
    </button>
  {:else}
    <button
      type="button"
      class="btn-outline w-full text-xs"
      onclick={onTrainEpoch}
      disabled={busy || !canTrain || batchesPerEpoch <= 0}
      title={canTrain
        ? `Train ${batchesPerEpoch} batches (1 epoch)`
        : 'Initialize the model first'}
    >
      Train 1 Epoch{batchesPerEpoch > 0 ? ` · ${batchesPerEpoch}` : ''}
    </button>
  {/if}

  {#if continuousRunning}
    <button
      type="button"
      class="btn-danger w-full text-xs"
      onclick={onStopContinuous}
      title="Stop continuous training at the next batch boundary"
    >
      Stop
    </button>
  {:else}
    <button
      type="button"
      class="btn-outline w-full text-xs"
      onclick={onTrainContinuously}
      disabled={busy || !canTrain}
      title={canTrain
        ? 'Train indefinitely until you click Stop'
        : 'Initialize the model first'}
    >
      Train Continuously
    </button>
  {/if}

  <hr class="my-2 border-[var(--color-border)]" />

  <h4
    class="text-[10px] font-semibold text-[var(--color-muted)] uppercase tracking-wide px-1"
  >
    Checkpoints
  </h4>

  <label class="flex flex-col gap-1 text-xs">
    <span class="text-[var(--color-muted)]">Checkpoint filename:</span>
    <input
      type="text"
      class="input !py-0.5 !px-2 !text-xs !w-full"
      placeholder="model.pt"
      value={training.checkpointFilename}
      oninput={(e) =>
        (training.checkpointFilename = (e.currentTarget as HTMLInputElement).value)}
    />
  </label>

  <div class="flex gap-1">
    <button
      type="button"
      class="btn-outline flex-1 text-xs"
      onclick={onLoadCheckpoint}
      disabled={busy || !canLoad}
      title={canLoad
        ? 'Load this checkpoint'
        : 'No checkpoint with that filename'}
    >
      Load
    </button>
    <button
      type="button"
      class="btn-outline flex-1 text-xs"
      onclick={onSaveCheckpoint}
      disabled={busy || !canTrain || !training.checkpointFilename.trim()}
      title={canTrain
        ? 'Save the current session under this filename'
        : 'Initialize the model first'}
    >
      Save
    </button>
  </div>
</aside>

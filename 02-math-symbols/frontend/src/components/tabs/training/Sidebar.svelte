<script lang="ts">
  import { training } from '../../../state.svelte';

  let {
    onReinitialize,
    onTrainBatch,
    onSaveCheckpoint,
    onLoadCheckpoint,
    canTrain,
    busy,
    canReinit,
    reinitBlockedReason,
  }: {
    onReinitialize: () => void;
    onTrainBatch: () => void;
    onSaveCheckpoint: () => void;
    onLoadCheckpoint: () => void;
    canTrain: boolean;
    busy: boolean;
    canReinit: boolean;
    reinitBlockedReason: string;
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
  <button
    type="button"
    class="btn-outline w-full text-xs"
    onclick={onReinitialize}
    disabled={busy || !canReinit}
    title={canReinit
      ? 'Build a fresh model from the current architecture and hyperparameters'
      : reinitBlockedReason}
  >
    Re-Initialize Model
  </button>
  {#if !canReinit}
    <div
      class="text-[10px] text-[var(--color-danger)] leading-snug px-1 -mt-1"
    >
      {reinitBlockedReason}
    </div>
  {/if}

  <button
    type="button"
    class="btn-primary w-full text-xs"
    onclick={onTrainBatch}
    disabled={busy || !canTrain}
    title={canTrain
      ? 'Run one forward + backward + optimizer step on the current batch'
      : 'Initialize the model first'}
  >
    Train (1 Batch)
  </button>

  <button
    type="button"
    class="btn-outline w-full text-xs"
    disabled
    title="Coming soon"
  >
    Train (1 Epoch)
  </button>

  <button
    type="button"
    class="btn-outline w-full text-xs"
    disabled
    title="Coming soon"
  >
    Train Continuously
  </button>

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

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
    onChangeFilename,
    onDeleteCheckpoint,
    onToggleAutoSave,
    onToggleAutoLoadOnRestart,
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
    onChangeFilename: (v: string) => void;
    onDeleteCheckpoint: (name: string) => void;
    onToggleAutoSave: (v: boolean) => void;
    onToggleAutoLoadOnRestart: (v: boolean) => void;
    canTrain: boolean;
    busy: boolean;
    canReinit: boolean;
    reinitBlockedReason: string;
    batchesPerEpoch: number;
    epochRunning: boolean;
    continuousRunning: boolean;
  } = $props();

  // Metadata for the file matching the current filename input. Null when
  // the input is empty or the file doesn't exist on disk yet — drives
  // both the "filesize / last save" line and the Load button's enabled
  // state, so the two can never disagree.
  let currentFileInfo = $derived.by(() => {
    const f = training.checkpointFilename.trim();
    if (!f) return null;
    const withExt = f.endsWith('.pt') ? f : `${f}.pt`;
    return (
      training.availableCheckpoints.find((c) => c.name === withExt) ?? null
    );
  });

  // Load is only enabled when the typed filename actually points to a
  // checkpoint file. The `length` read tags this derived as a dep of
  // the array so it re-runs on the next refresh after a save/list call.
  let canLoad = $derived(
    training.availableCheckpoints.length >= 0 && currentFileInfo !== null
  );

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  }

  function formatTimeFull(unixSeconds: number): string {
    const d = new Date(unixSeconds * 1000);
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  // Compact mtime format used by the "last save" line under the input:
  // HH:MM if today, "Apr 27" if this year, "27/4/25" otherwise.
  function formatTimeCompact(unixSeconds: number): string {
    const d = new Date(unixSeconds * 1000);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) {
      return d.toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
      });
    }
    if (d.getFullYear() === now.getFullYear()) {
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
      });
    }
    return d.toLocaleDateString(undefined, {
      year: '2-digit',
      month: 'numeric',
      day: 'numeric',
    });
  }
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
        ? training.hasSession
          ? 'Replace the current model with a fresh one built from the architecture and hyperparameters'
          : 'Build a model from the current architecture and hyperparameters'
        : reinitBlockedReason}
    >
      {training.hasSession ? 'Re-Initialize Model' : 'Initialize Model'}
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
        onChangeFilename((e.currentTarget as HTMLInputElement).value)}
    />
  </label>

  <div
    class="text-[10px] text-[var(--color-muted)] leading-snug px-1 -mt-1
           font-mono break-all"
    title={currentFileInfo ? formatTimeFull(currentFileInfo.mtime) : undefined}
  >
    {#if currentFileInfo}
      filesize {formatBytes(currentFileInfo.size)}<br />
      last save: {formatTimeCompact(currentFileInfo.mtime)}
    {:else if training.checkpointFilename.trim()}
      not saved yet
    {:else}
      &nbsp;
    {/if}
  </div>

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

  <label class="flex items-center gap-2 text-xs px-1 cursor-pointer">
    <input
      type="checkbox"
      class="cursor-pointer"
      checked={training.autoSave}
      onchange={(e) =>
        onToggleAutoSave((e.currentTarget as HTMLInputElement).checked)}
    />
    <span>Auto-save</span>
  </label>

  <label class="flex items-center gap-2 text-xs px-1 cursor-pointer">
    <input
      type="checkbox"
      class="cursor-pointer"
      checked={training.autoLoadOnRestart}
      onchange={(e) =>
        onToggleAutoLoadOnRestart(
          (e.currentTarget as HTMLInputElement).checked
        )}
    />
    <span>Auto-load on restart</span>
  </label>

  {#if training.availableCheckpoints.length > 0}
    <h5
      class="text-[10px] font-semibold text-[var(--color-muted)] uppercase
             tracking-wide px-1 mt-1"
    >
      Saved files
    </h5>
    <ul class="flex flex-col text-[11px] font-mono">
      {#each training.availableCheckpoints as f (f.name)}
        {@const isCurrent =
          (training.checkpointFilename.endsWith('.pt')
            ? training.checkpointFilename
            : `${training.checkpointFilename}.pt`) === f.name}
        <li
          class="ckpt-row"
          class:ckpt-row-active={isCurrent}
          title={`${f.name}\nfilesize ${formatBytes(f.size)}\nlast save: ${formatTimeFull(f.mtime)}`}
        >
          <button
            type="button"
            class="ckpt-name"
            onclick={() => onChangeFilename(f.name)}
          >
            {f.name}
          </button>
          <button
            type="button"
            class="ckpt-delete"
            aria-label={`Delete ${f.name}`}
            title={`Delete ${f.name}`}
            onclick={(e) => {
              e.stopPropagation();
              onDeleteCheckpoint(f.name);
            }}
          >
            ×
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</aside>

<style>
  .ckpt-row {
    display: flex;
    align-items: stretch;
    border-radius: 0.25rem;
    transition: background 100ms ease;
  }
  .ckpt-row:hover {
    background: var(--color-surface-2);
  }
  .ckpt-row-active {
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .ckpt-row-active:hover {
    background: color-mix(in srgb, var(--color-accent) 18%, transparent);
  }
  .ckpt-name {
    flex: 1;
    min-width: 0;
    text-align: left;
    background: none;
    border: 0;
    padding: 0.2rem 0.5rem;
    color: var(--color-text);
    font: inherit;
    cursor: pointer;
    /* Truncate long filenames with ellipsis instead of wrapping. */
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .ckpt-delete {
    flex-shrink: 0;
    width: 1.4rem;
    background: none;
    border: 0;
    padding: 0;
    color: var(--color-muted);
    font-size: 0.95rem;
    line-height: 1;
    cursor: pointer;
    border-radius: 0.25rem;
    transition: background 100ms ease, color 100ms ease;
  }
  .ckpt-delete:hover {
    background: color-mix(in srgb, var(--color-danger) 18%, transparent);
    color: var(--color-danger);
  }
</style>

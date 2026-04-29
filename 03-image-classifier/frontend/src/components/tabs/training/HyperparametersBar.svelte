<script lang="ts">
  import { architecture, dataset, training } from '../../../state.svelte';

  const OPTIMIZERS = [
    { id: 'adam', label: 'Adam' },
    { id: 'adamw', label: 'AdamW' },
    { id: 'sgd', label: 'SGD' },
  ] as const;

  // Validation cadence: how often (in train batches) we sweep through the
  // val pipeline to log a point on the validation curve. Tuned visually:
  // every 10 is the default — frequent enough to see overfitting in real
  // time, but not so frequent that val sweeps dominate wall-clock time.
  let batchesPerEpoch = $derived.by(() => {
    const totalTrain = dataset.status?.num_train ?? 0;
    const bs = architecture.hyperparameters.batch_size;
    if (totalTrain <= 0 || bs <= 0) return 0;
    return Math.max(1, Math.ceil(totalTrain / bs));
  });
</script>

<div class="flex items-center gap-4 flex-wrap text-sm">
  <label class="flex items-center gap-1.5">
    <span class="text-xs text-[var(--color-muted)]">Learning rate</span>
    <input
      type="number"
      step="0.0001"
      min="0"
      class="input !py-0.5 !px-2 !text-xs !w-24"
      value={architecture.hyperparameters.lr}
      oninput={(e) =>
        (architecture.hyperparameters.lr = parseFloat(
          (e.currentTarget as HTMLInputElement).value
        ))}
    />
  </label>

  <label class="flex items-center gap-1.5">
    <span class="text-xs text-[var(--color-muted)]">Batch size</span>
    <input
      type="number"
      step="1"
      min="1"
      class="input !py-0.5 !px-2 !text-xs !w-20"
      value={architecture.hyperparameters.batch_size}
      oninput={(e) =>
        (architecture.hyperparameters.batch_size = parseInt(
          (e.currentTarget as HTMLInputElement).value,
          10
        ))}
    />
  </label>

  <label class="flex items-center gap-1.5">
    <span class="text-xs text-[var(--color-muted)]">Optimizer</span>
    <select
      class="input !py-0.5 !px-2 !text-xs !font-sans !w-auto"
      value={architecture.hyperparameters.optimizer}
      onchange={(e) =>
        (architecture.hyperparameters.optimizer =
          (e.currentTarget as HTMLSelectElement)
            .value as typeof architecture.hyperparameters.optimizer)}
    >
      {#each OPTIMIZERS as o}
        <option value={o.id}>{o.label}</option>
      {/each}
    </select>
  </label>

  <label
    class="flex items-center gap-1.5"
    title="How often (in train batches) to evaluate on a held-out validation batch."
  >
    <span class="text-xs text-[var(--color-muted)]">Validate every N</span>
    <input
      type="number"
      step="1"
      min="1"
      class="input !py-0.5 !px-2 !text-xs !w-16"
      value={training.validateEveryN}
      oninput={(e) => {
        const v = parseInt((e.currentTarget as HTMLInputElement).value, 10);
        if (Number.isFinite(v) && v >= 1) training.validateEveryN = v;
      }}
    />
  </label>

  <span class="text-[10px] text-[var(--color-muted)] font-mono ml-auto">
    1 epoch ≈ {batchesPerEpoch.toLocaleString()} batches
  </span>
</div>

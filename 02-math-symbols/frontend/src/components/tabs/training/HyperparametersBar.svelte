<script lang="ts">
  import { architecture } from '../../../state.svelte';

  const OPTIMIZERS = [
    { id: 'adam', label: 'Adam' },
    { id: 'adamw', label: 'AdamW' },
    { id: 'sgd', label: 'SGD' },
  ] as const;
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
</div>

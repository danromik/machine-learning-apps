<script lang="ts">
  import { algorithm, ui } from '../../state.svelte';

  // Value iteration's tunable knobs (label per key).
  const HP_FIELDS: { key: string; label: string; hint?: string }[] = [
    { key: 'lr', label: 'Learning rate' },
    { key: 'batch_size', label: 'Batch size', hint: 'scrambled states per gradient step' },
    { key: 'hidden', label: 'Hidden width', hint: 'first MLP layer (second = half)' },
    { key: 'target_update', label: 'Target sync (batches)' },
    { key: 'weight_decay', label: 'Weight decay' },
  ];

  function resetDefaults() {
    const info = algorithm.catalog.find((a) => a.id === algorithm.algo);
    if (info) algorithm.hyperparameters = { ...info.default_hyperparameters };
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-4xl mx-auto flex flex-col gap-6">
    <div>
      <h2 class="text-lg font-bold text-[var(--color-heading)]">Algorithm</h2>
      <p class="text-sm text-[var(--color-muted)] mt-1">
        How the agent learns. The cube's known model lets us skip sparse-reward
        exploration entirely and instead regress a cost-to-go heuristic.
      </p>
    </div>

    <!-- Algorithm card(s) -->
    <div class="grid gap-3" style="grid-template-columns: 1fr">
      {#each algorithm.catalog as algo}
        <button
          type="button"
          class="card p-4 text-left transition-colors"
          class:selected={algorithm.algo === algo.id}
          onclick={() => (algorithm.algo = algo.id)}
        >
          <div class="flex items-center justify-between mb-1">
            <span class="font-semibold text-[var(--color-heading)] text-sm">{algo.label}</span>
          </div>
          <p class="text-xs text-[var(--color-muted)] leading-relaxed">{algo.description}</p>
        </button>
      {/each}
    </div>

    <!-- Hyperparameters -->
    <div class="card p-4">
      <div class="flex items-center justify-between mb-3">
        <span class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide"
          >Hyperparameters</span
        >
        <button type="button" class="btn-ghost text-xs" onclick={resetDefaults}
          >Reset to defaults</button
        >
      </div>
      <div class="grid gap-3" style="grid-template-columns: repeat(2, 1fr)">
        {#each HP_FIELDS as f}
          {#if algorithm.hyperparameters[f.key] !== undefined}
            <label class="flex flex-col gap-0.5 text-sm">
              <span class="flex items-center justify-between gap-3">
                <span class="text-[var(--color-text)]">{f.label}</span>
                <input
                  type="number"
                  class="input w-28 text-right font-mono"
                  step="any"
                  value={algorithm.hyperparameters[f.key]}
                  oninput={(e) =>
                    (algorithm.hyperparameters[f.key] = +(e.target as HTMLInputElement).value)}
                />
              </span>
              {#if f.hint}<span class="text-[10px] text-[var(--color-muted)]">{f.hint}</span>{/if}
            </label>
          {/if}
        {/each}
      </div>
      <p class="text-xs text-[var(--color-muted)] mt-3">
        Learning rate is hot-swappable while training; the rest take effect when you
        next initialize the agent (or start a background run).
      </p>
    </div>

    <div>
      <button type="button" class="btn-primary" onclick={() => (ui.activeTab = 'training')}>
        Go to training →
      </button>
    </div>
  </div>
</div>

<style>
  .selected {
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 8%, transparent);
  }
</style>

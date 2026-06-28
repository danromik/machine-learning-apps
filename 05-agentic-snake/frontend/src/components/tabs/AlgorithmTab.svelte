<script lang="ts">
  import { algorithm, environment, ui } from '../../state.svelte';
  import type { AlgorithmId } from '../../api';

  // Tabular Q-learning needs a small discrete state — it can't work on the
  // full grid (too many states for a lookup table). Disable it in grid mode.
  let gridMode = $derived(environment.observation === 'grid');
  function disabledFor(id: AlgorithmId): boolean {
    return gridMode && id === 'qlearning';
  }

  // Which hyperparameters are meaningful for each algorithm (and a label).
  const HP_FIELDS: Record<AlgorithmId, { key: string; label: string }[]> = {
    qlearning: [
      { key: 'lr', label: 'Learning rate' },
      { key: 'gamma', label: 'Discount γ' },
      { key: 'epsilon_start', label: 'ε start' },
      { key: 'epsilon_min', label: 'ε min' },
      { key: 'epsilon_decay', label: 'ε decay / episode' },
    ],
    dqn: [
      { key: 'lr', label: 'Learning rate' },
      { key: 'gamma', label: 'Discount γ' },
      { key: 'epsilon_start', label: 'ε start' },
      { key: 'epsilon_min', label: 'ε min' },
      { key: 'epsilon_decay', label: 'ε decay / episode' },
      { key: 'hidden', label: 'Hidden width' },
      { key: 'batch_size', label: 'Batch size' },
      { key: 'buffer_size', label: 'Replay buffer' },
      { key: 'target_update', label: 'Target sync (steps)' },
      { key: 'warmup', label: 'Warmup steps' },
    ],
    reinforce: [
      { key: 'lr', label: 'Learning rate' },
      { key: 'gamma', label: 'Discount γ' },
      { key: 'hidden', label: 'Hidden width' },
    ],
  };

  function selectAlgo(id: AlgorithmId) {
    if (algorithm.algo === id || disabledFor(id)) return;
    algorithm.algo = id;
    const info = algorithm.catalog.find((a) => a.id === id);
    if (info) algorithm.hyperparameters = { ...info.default_hyperparameters };
  }

  // If the user switches to the grid observation while Q-learning is selected,
  // move them to DQN (the closest deep equivalent) so they can't get stuck on
  // an invalid combo.
  $effect(() => {
    if (gridMode && algorithm.algo === 'qlearning') selectAlgo('dqn');
  });

  function resetDefaults() {
    const info = algorithm.catalog.find((a) => a.id === algorithm.algo);
    if (info) algorithm.hyperparameters = { ...info.default_hyperparameters };
  }

  let fields = $derived(HP_FIELDS[algorithm.algo] ?? []);
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-4xl mx-auto flex flex-col gap-6">
    <div>
      <h2 class="text-lg font-bold text-[var(--color-heading)]">Algorithm</h2>
      <p class="text-sm text-[var(--color-muted)] mt-1">
        How the agent learns. All three share the same three actions (turn left
        / go straight / turn right) and the
        <strong>{environment.observation === 'grid' ? 'full-grid' : '11-feature'}</strong>
        observation you chose on the Environment tab.
      </p>
    </div>

    {#if gridMode}
      <div
        class="text-xs rounded-md px-3 py-2 border border-[var(--color-border)]
               bg-[var(--color-surface)]/60 text-[var(--color-muted)]"
      >
        <strong class="text-[var(--color-text)]">Grid observation selected.</strong>
        Tabular Q-learning is unavailable — the full board has far too many states
        for a lookup table, which is exactly why deep RL uses a neural network to
        approximate the value/policy. Choose DQN or REINFORCE.
      </div>
    {/if}

    <!-- Algorithm cards -->
    <div class="grid gap-3" style="grid-template-columns: repeat(3, 1fr)">
      {#each algorithm.catalog as algo}
        {@const isDisabled = disabledFor(algo.id)}
        <button
          type="button"
          class="card p-4 text-left transition-colors"
          class:selected={algorithm.algo === algo.id}
          class:opacity-40={isDisabled}
          class:cursor-not-allowed={isDisabled}
          disabled={isDisabled}
          title={isDisabled ? 'Not available with the full-grid observation' : ''}
          onclick={() => selectAlgo(algo.id)}
        >
          <div class="flex items-center justify-between mb-1">
            <span class="font-semibold text-[var(--color-heading)] text-sm">{algo.label}</span>
            {#if isDisabled}
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-border)]/40 text-[var(--color-muted)]"
                >grid: n/a</span
              >
            {:else if !algo.uses_network}
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-border)]/40 text-[var(--color-muted)]"
                >no NN</span
              >
            {/if}
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
        <button type="button" class="btn-ghost text-xs" onclick={resetDefaults}>Reset to defaults</button>
      </div>
      <div class="grid gap-3" style="grid-template-columns: repeat(2, 1fr)">
        {#each fields as f}
          {#if algorithm.hyperparameters[f.key] !== undefined}
            <label class="flex items-center justify-between gap-3 text-sm">
              <span class="text-[var(--color-text)]">{f.label}</span>
              <input
                type="number"
                class="input w-28 text-right font-mono"
                step="any"
                value={algorithm.hyperparameters[f.key]}
                oninput={(e) =>
                  (algorithm.hyperparameters[f.key] = +(e.target as HTMLInputElement).value)}
              />
            </label>
          {/if}
        {/each}
      </div>
      <p class="text-xs text-[var(--color-muted)] mt-3">
        Learning rate is hot-swappable while training; the rest take effect when
        you next initialize the agent.
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

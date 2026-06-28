<script lang="ts">
  import { environment, ui } from '../../state.svelte';
  import GameBoard from '../GameBoard.svelte';
  import type { Frame } from '../../api';

  // A static preview frame: a length-3 snake centered, with a food cell, so
  // the user sees the grid scale as they change width/height.
  let previewFrame = $derived.by((): Frame => {
    const cx = Math.floor(environment.width / 2);
    const cy = Math.floor(environment.height / 2);
    const snake: [number, number][] = [
      [cx, cy],
      [cx - 1, cy],
      [cx - 2, cy],
    ].filter(([x]) => x >= 0) as [number, number][];
    const food: [number, number] = [
      Math.min(environment.width - 1, cx + 2),
      Math.max(0, cy - 2),
    ];
    return {
      width: environment.width,
      height: environment.height,
      snake,
      food,
      score: 0,
      steps: 0,
      done: false,
    };
  });

  const OBSERVATIONS: {
    id: 'features' | 'grid';
    title: string;
    blurb: string;
    detail: string;
  }[] = [
    {
      id: 'features',
      title: 'Engineered features',
      blurb: '11-value vector',
      detail:
        'Danger in the 3 cells next to the head, current heading, and the ' +
        "food's direction. Tiny and fast; works with all three algorithms — " +
        'but the agent only sees its immediate surroundings, so it is blind to ' +
        'the shape of its own body and eventually traps itself.',
    },
    {
      id: 'grid',
      title: 'Full grid',
      blurb: 'whole board (CNN)',
      detail:
        'The entire board as an image (snake body, head, food), fed to a small ' +
        'convolutional network. The agent can finally see its whole body and ' +
        'avoid trapping itself — but it learns slower, needs more episodes, and ' +
        'tabular Q-learning no longer applies (only DQN / REINFORCE).',
    },
  ];

  const REWARDS: { key: keyof typeof environment.reward; label: string; hint: string; min: number; max: number; step: number }[] = [
    { key: 'food', label: 'Eat food', hint: 'reward for eating an apple', min: 0, max: 5, step: 0.1 },
    { key: 'death', label: 'Die', hint: 'penalty for hitting a wall or itself', min: -5, max: 0, step: 0.1 },
    { key: 'step', label: 'Per step', hint: 'living cost each step — negative discourages stalling', min: -0.1, max: 0.1, step: 0.005 },
    { key: 'toward_food', label: 'Toward food', hint: 'bonus for moving closer to food (shaping)', min: 0, max: 0.2, step: 0.005 },
    { key: 'away_from_food', label: 'Away from food', hint: 'penalty for moving farther (use a negative value)', min: -0.2, max: 0, step: 0.005 },
  ];
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-5xl mx-auto grid gap-6" style="grid-template-columns: 1fr 1fr">
    <!-- Left: controls -->
    <div class="flex flex-col gap-5">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Environment</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          The grid the snake lives on and the reward it learns from. Changes
          apply when you next initialize the agent on the Training tab.
        </p>
      </div>

      <div class="card p-4 flex flex-col gap-3">
        <span class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide"
          >Observation model</span
        >
        <p class="text-xs text-[var(--color-muted)]">
          What the agent gets to see each step — the data its policy is a
          function of.
        </p>
        <div class="grid gap-2" style="grid-template-columns: 1fr 1fr">
          {#each OBSERVATIONS as obs}
            <button
              type="button"
              class="card p-3 text-left transition-colors"
              class:obs-selected={environment.observation === obs.id}
              onclick={() => (environment.observation = obs.id)}
            >
              <div class="flex items-center justify-between mb-0.5">
                <span class="font-semibold text-sm text-[var(--color-heading)]">{obs.title}</span>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-border)]/40 text-[var(--color-muted)]"
                  >{obs.blurb}</span
                >
              </div>
              <p class="text-xs text-[var(--color-muted)] leading-relaxed">{obs.detail}</p>
            </button>
          {/each}
        </div>
        {#if environment.observation === 'grid'}
          <p class="text-xs text-[var(--color-muted)] leading-relaxed">
            Grid mode disables tabular Q-learning on the Algorithm tab. Give DQN
            or REINFORCE more episodes than you would in feature mode — the CNN
            has more to learn.
          </p>
        {/if}
      </div>

      <div class="card p-4 flex flex-col gap-4">
        <span class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide">Grid size</span>
        {#each [{ k: 'width', label: 'Width' }, { k: 'height', label: 'Height' }] as dim}
          <label class="flex items-center gap-3 text-sm">
            <span class="w-16 text-[var(--color-text)]">{dim.label}</span>
            <input
              type="range"
              min="6"
              max="20"
              step="1"
              class="flex-1"
              value={environment[dim.k as 'width' | 'height']}
              oninput={(e) =>
                (environment[dim.k as 'width' | 'height'] = +(e.target as HTMLInputElement).value)}
            />
            <span class="w-8 text-right font-mono text-[var(--color-accent)]"
              >{environment[dim.k as 'width' | 'height']}</span
            >
          </label>
        {/each}
        <p class="text-xs text-[var(--color-muted)]">
          Smaller grids learn faster; larger grids are harder and allow higher scores.
        </p>
      </div>

      <div class="card p-4 flex flex-col gap-3">
        <span class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide">Reward</span>
        {#each REWARDS as r}
          <label class="flex items-center gap-3 text-sm">
            <span class="w-28 text-[var(--color-text)]" title={r.hint}>{r.label}</span>
            <input
              type="range"
              min={r.min}
              max={r.max}
              step={r.step}
              class="flex-1"
              value={environment.reward[r.key]}
              oninput={(e) => (environment.reward[r.key] = +(e.target as HTMLInputElement).value)}
            />
            <span class="w-12 text-right font-mono text-[var(--color-accent)]"
              >{environment.reward[r.key].toFixed(3)}</span
            >
          </label>
        {/each}
        <p class="text-xs text-[var(--color-muted)] leading-relaxed">
          The defaults (+1 eat, −1 die, no shaping) are a clean starting point.
          Add a little <em>toward food</em> shaping to speed up early learning —
          but watch for the agent gaming the shaping instead of actually eating.
        </p>
      </div>

      <div>
        <button type="button" class="btn-primary" onclick={() => (ui.activeTab = 'algorithm')}>
          Choose an algorithm →
        </button>
      </div>
    </div>

    <!-- Right: live preview -->
    <div class="flex flex-col items-center gap-3">
      <span class="text-xs font-semibold text-[var(--color-muted)] uppercase tracking-wide self-start"
        >Preview</span
      >
      <GameBoard frame={previewFrame} size={360} />
      <p class="text-xs text-[var(--color-muted)] text-center max-w-xs">
        The snake (green, brightest at the head) eats food (red) to grow. Hitting
        a wall or its own body ends the game.
      </p>
    </div>
  </div>
</div>

<style>
  .obs-selected {
    border-color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 8%, transparent);
  }
</style>

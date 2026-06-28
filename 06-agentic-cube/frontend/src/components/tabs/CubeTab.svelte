<script lang="ts">
  import { cube, ui } from '../../state.svelte';
  import { solvedFrame } from '../../cubeGeometry';
  import CubeView3D from '../CubeView3D.svelte';

  let view: CubeView3D | null = $state(null);

  // Rebuild the 3D preview whenever the cube size changes.
  $effect(() => {
    const sz = cube.size;
    if (view) view.reset(solvedFrame(sz));
  });

  function setSize(s: 2 | 3) {
    cube.size = s;
    // Keep the curriculum ceiling sensible for the chosen cube.
    cube.curriculum.maxK = s === 2 ? 14 : 20;
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-6xl mx-auto grid gap-6" style="grid-template-columns: 360px 1fr">
    <!-- Controls -->
    <div class="flex flex-col gap-4">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Cube & Curriculum</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          Choose the cube and how the reverse-scramble curriculum ramps difficulty.
        </p>
      </div>

      <div class="card p-4 flex flex-col gap-3">
        <span class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)]"
          >Cube size</span
        >
        <div class="flex gap-2">
          <button
            type="button"
            class="flex-1 {cube.size === 2 ? 'btn-primary' : 'btn-outline'}"
            onclick={() => setSize(2)}
          >
            2×2×2
          </button>
          <button
            type="button"
            class="flex-1 {cube.size === 3 ? 'btn-primary' : 'btn-outline'}"
            onclick={() => setSize(3)}
          >
            3×3×3
          </button>
        </div>
        <p class="text-xs text-[var(--color-muted)]">
          {#if cube.size === 2}
            Pocket Cube — ~3.6M states. Trains to a full solver in minutes.
          {:else}
            Standard cube — ~4.3×10¹⁹ states. Best-effort: learns shallow scrambles
            fast, pushes deeper over a long run.
          {/if}
        </p>
      </div>

      <div class="card p-4 flex flex-col gap-4">
        <span class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)]"
          >Reverse-scramble curriculum</span
        >

        <label class="flex flex-col gap-1 text-sm">
          <span class="flex justify-between">
            <span>Start depth (k)</span>
            <span class="font-mono text-[var(--color-muted)]">{cube.curriculum.startK}</span>
          </span>
          <input type="range" min="1" max="8" bind:value={cube.curriculum.startK} />
        </label>

        <label class="flex flex-col gap-1 text-sm">
          <span class="flex justify-between">
            <span>Max depth (k)</span>
            <span class="font-mono text-[var(--color-muted)]">{cube.curriculum.maxK}</span>
          </span>
          <input type="range" min="2" max="26" bind:value={cube.curriculum.maxK} />
        </label>

        <label class="flex flex-col gap-1 text-sm">
          <span class="flex justify-between">
            <span>Promote at solve-rate</span>
            <span class="font-mono text-[var(--color-muted)]"
              >{(cube.curriculum.promoteAt * 100).toFixed(0)}%</span
            >
          </span>
          <input
            type="range"
            min="0.5"
            max="1"
            step="0.05"
            bind:value={cube.curriculum.promoteAt}
          />
        </label>
        <p class="text-xs text-[var(--color-muted)]">
          Training starts at depth {cube.curriculum.startK} and ramps up to
          {cube.curriculum.maxK} — advancing a level each time the solver clears
          {(cube.curriculum.promoteAt * 100).toFixed(0)}% at the current depth.
        </p>
      </div>

      <button type="button" class="btn-primary" onclick={() => (ui.activeTab = 'algorithm')}>
        Next: Algorithm →
      </button>
    </div>

    <!-- 3D preview -->
    <div class="card p-3 flex flex-col min-h-[420px]">
      <div class="text-xs font-semibold text-[var(--color-heading)] mb-1">
        {cube.size}×{cube.size}×{cube.size} cube — drag to rotate, scroll to zoom
      </div>
      <div class="flex-1 min-h-[380px]">
        <CubeView3D bind:this={view} size={cube.size} />
      </div>
    </div>
  </div>
</div>

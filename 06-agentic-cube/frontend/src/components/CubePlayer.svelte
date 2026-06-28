<!--
  A 3D cube viewer with playback transport (play/pause/restart, fps, moves
  slider) over a SolveResult. Shared by the Watch tab and the Training tab's
  "Play last training round" view so the playback logic lives in one place.

  `stepIndex` is bindable so a parent can read the current move (e.g. the Watch
  tab's per-step heuristic overlay).
-->
<script lang="ts">
  import CubeView3D from './CubeView3D.svelte';
  import { solvedFrame } from '../cubeGeometry';
  import { playSolve, playTurn, playFail, resumeAudio } from '../sound';
  import type { SolveResult } from '../api';

  let {
    result = null,
    size = 3,
    stepIndex = $bindable(0),
    height = '300px',
  }: {
    result?: SolveResult | null;
    size?: number;
    stepIndex?: number;
    height?: string;
  } = $props();

  let view: CubeView3D | null = $state(null);
  let playing = $state(false);
  let fps = $state(2);

  // Reset the cube whenever a new result is loaded (or cleared).
  let lastResult: SolveResult | null = null;
  $effect(() => {
    void result; // track
    if (result !== lastResult) {
      lastResult = result;
      playing = false;
      stepIndex = 0;
      if (view) view.reset(result ? result.frames[0] : solvedFrame(size));
    }
  });

  // Show a solved cube of the right size when there's nothing to play yet.
  $effect(() => {
    const sz = size;
    if (view && !result) view.reset(solvedFrame(sz));
  });

  async function play() {
    if (!result || playing) return;
    resumeAudio(); // unlock audio within the click gesture (autoplay policy)
    const total = result.steps.length;
    if (stepIndex >= total) {
      stepIndex = 0;
      view?.reset(result.frames[0]);
    }
    playing = true;
    while (playing && result && stepIndex < result.steps.length) {
      const dur = Math.max(120, 1000 / fps);
      playTurn(); // a clack as the layer turns
      await view?.animateMove(result.steps[stepIndex].move, dur);
      stepIndex += 1;
    }
    // Played through to the end (not paused mid-way): reward or commiserate.
    if (result && stepIndex >= total) {
      if (result.solved) playSolve();
      else playFail();
    }
    playing = false;
  }

  function pause() {
    playing = false;
  }

  function scrubTo(idx: number) {
    if (!result) return;
    playing = false;
    stepIndex = Math.max(0, Math.min(result.frames.length - 1, idx));
    view?.reset(result.frames[stepIndex]);
  }
</script>

<div class="flex flex-col gap-2 min-h-0">
  <div style="height: {height}">
    <CubeView3D bind:this={view} {size} />
  </div>
  <div class="flex items-center gap-2">
    {#if playing}
      <button type="button" class="btn-outline text-sm" onclick={pause}>Pause</button>
    {:else}
      <button type="button" class="btn-primary text-sm" disabled={!result} onclick={play}>Play</button>
    {/if}
    <button type="button" class="btn-ghost text-sm" disabled={!result} onclick={() => scrubTo(0)}
      >Restart</button
    >
    <div class="flex items-center gap-1.5 text-xs text-[var(--color-muted)] ml-1">
      <span>fps</span>
      <input type="range" min="1" max="8" bind:value={fps} class="w-20" />
      <span class="font-mono w-4">{fps}</span>
    </div>
  </div>
  {#if result}
    <input
      type="range"
      min="0"
      max={result.frames.length - 1}
      value={stepIndex}
      oninput={(e) => scrubTo(+(e.target as HTMLInputElement).value)}
    />
    <div class="text-xs font-mono text-[var(--color-muted)]">
      move {stepIndex} / {result.steps.length}{result.solved ? ' · ✓ solved' : ''}
    </div>
  {/if}
</div>

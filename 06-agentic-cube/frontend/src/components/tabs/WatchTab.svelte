<script lang="ts">
  import { cube, training, ui } from '../../state.svelte';
  import { api, type SolveResult } from '../../api';
  import CubePlayer from '../CubePlayer.svelte';

  let result = $state<SolveResult | null>(null);
  let stepIndex = $state(0);
  let busy = $state(false);
  let scrambleK = $state(8);
  let msg = $state('');

  let curStep = $derived(
    result && stepIndex > 0 && stepIndex <= result.steps.length
      ? result.steps[stepIndex - 1]
      : null
  );
  let nextStep = $derived(
    result && stepIndex < result.steps.length ? result.steps[stepIndex] : null
  );

  async function scrambleAndSolve() {
    busy = true;
    msg = '';
    try {
      let r: SolveResult;
      try {
        r = await api.play(scrambleK);
      } catch {
        // No live session — try a bundled/pretrained checkpoint for this size.
        const fname = `pretrained-${cube.size}x${cube.size}.pt`;
        try {
          await api.loadCheckpoint(fname);
          r = await api.play(scrambleK);
          msg = `loaded ${fname}`;
        } catch {
          msg =
            'No trained agent yet — initialize and train on the Training tab, ' +
            'or load a checkpoint.';
          busy = false;
          return;
        }
      }
      result = r;
    } catch (e) {
      msg = `solve failed: ${e}`;
    } finally {
      busy = false;
    }
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-6xl mx-auto grid gap-6" style="grid-template-columns: 1fr 320px">
    <!-- 3D cube + transport -->
    <div class="card p-3 flex flex-col">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs font-semibold text-[var(--color-heading)]"
          >Watch the solver — drag to rotate, scroll to zoom</span
        >
        {#if result}
          <span class="text-xs font-mono text-[var(--color-muted)]">
            {result.solved ? '✓ solved' : '✗ unsolved'} · {result.solution_len} moves
          </span>
        {/if}
      </div>
      <CubePlayer {result} size={cube.size} bind:stepIndex height="420px" />
    </div>

    <!-- Controls + overlay -->
    <div class="flex flex-col gap-4">
      <div class="card p-4 flex flex-col gap-3">
        <span class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)]"
          >Scramble & solve</span
        >
        <label class="flex flex-col gap-1 text-sm">
          <span class="flex justify-between">
            <span>Scramble depth (k)</span>
            <span class="font-mono text-[var(--color-muted)]">{scrambleK}</span>
          </span>
          <input type="range" min="1" max={cube.curriculum.maxK} bind:value={scrambleK} />
        </label>
        <button type="button" class="btn-primary" disabled={busy} onclick={scrambleAndSolve}>
          {busy ? 'Solving…' : 'Scramble & Solve'}
        </button>
        {#if msg}
          <p class="text-xs text-[var(--color-muted)]">{msg}</p>
        {/if}
        {#if !training.hasSession && !result}
          <p class="text-xs text-[var(--color-muted)]">
            Tip: train an agent on the
            <button class="underline" onclick={() => (ui.activeTab = 'training')}>Training tab</button>
            first.
          </p>
        {/if}
      </div>

      <!-- Per-step heuristic overlay -->
      {#if result}
        <div class="card p-4 flex flex-col gap-2">
          <span class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)]"
            >What the agent "thinks"</span
          >
          {#if curStep}
            <div class="text-sm">
              Last move: <span class="font-mono font-bold text-[var(--color-accent)]"
                >{curStep.move.name}</span
              >
            </div>
          {/if}
          {#if nextStep?.scores}
            <div class="text-[11px] text-[var(--color-muted)] mb-1">
              Predicted cost-to-go after each move (lower = closer to solved):
            </div>
            <div class="flex flex-col gap-1">
              {#each nextStep.scores.values as v, i}
                {@const minv = Math.min(...nextStep.scores.values)}
                {@const maxv = Math.max(...nextStep.scores.values, minv + 0.001)}
                <div class="flex items-center gap-2 text-xs">
                  <span class="w-8 font-mono text-[var(--color-muted)]"
                    >{result.move_catalog[i]?.name ?? i}</span
                  >
                  <div class="flex-1 h-3 rounded bg-[var(--color-border)]/40 overflow-hidden">
                    <div
                      class="h-full rounded"
                      style="width: {(100 * (1 - (v - minv) / (maxv - minv))).toFixed(0)}%;
                             background: {v === minv ? 'var(--color-accent)' : 'var(--color-muted)'}"
                    ></div>
                  </div>
                  <span class="w-12 text-right font-mono">{v.toFixed(2)}</span>
                </div>
              {/each}
            </div>
          {/if}
          <div class="text-[11px] text-[var(--color-muted)] mt-1">
            Scramble: {result.scramble_moves.join(' ')}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

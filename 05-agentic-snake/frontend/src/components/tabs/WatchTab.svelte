<script lang="ts">
  import { training } from '../../state.svelte';
  import { api, ACTION_NAMES, type PlayResult } from '../../api';
  import GameBoard from '../GameBoard.svelte';

  let result = $state<PlayResult | null>(null);
  let idx = $state(0);
  let playing = $state(false);
  let speed = $state(10); // frames per second
  let loading = $state(false);
  let error = $state('');
  let timer: ReturnType<typeof setTimeout> | null = null;

  let frame = $derived(result ? result.frames[Math.min(idx, result.frames.length - 1)] : null);
  // The step (action + scores) that produced the current frame.
  let step = $derived(result && idx > 0 ? result.steps[idx - 1] : null);

  function clearTimer() {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  }

  function advance() {
    if (!result) return;
    if (idx >= result.frames.length - 1) {
      playing = false;
      clearTimer();
      return;
    }
    idx += 1;
    timer = setTimeout(advance, 1000 / speed);
  }

  function play() {
    if (!result) return;
    if (idx >= result.frames.length - 1) idx = 0;
    playing = true;
    clearTimer();
    timer = setTimeout(advance, 1000 / speed);
  }

  function pause() {
    playing = false;
    clearTimer();
  }

  async function loadGame(greedy: boolean) {
    if (!training.hasSession) return;
    pause();
    loading = true;
    error = '';
    try {
      result = await api.play(greedy);
      idx = 0;
      play();
    } catch (e) {
      error = `${e}`;
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    return () => clearTimer();
  });

  let kind = $derived(step?.scores?.kind ?? null);
  let maxScore = $derived(step?.scores ? Math.max(...step.scores.values.map(Math.abs), 1e-6) : 1);
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-5xl mx-auto flex flex-col gap-4">
    <div class="flex items-end justify-between gap-4 flex-wrap">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Watch the agent play</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          A single greedy game, frame by frame, with the agent's per-action
          scores at each step.
        </p>
      </div>
      <div class="flex gap-2">
        <button
          type="button"
          class="btn-primary"
          disabled={!training.hasSession || loading}
          onclick={() => loadGame(true)}>New greedy game</button
        >
        <button
          type="button"
          class="btn-outline"
          disabled={!training.hasSession || loading}
          onclick={() => loadGame(false)}
          title="Let the agent explore (sample actions) instead of always taking the best"
          >Exploring</button
        >
      </div>
    </div>

    {#if !training.hasSession}
      <div class="card p-8 text-center text-[var(--color-muted)]">
        No agent yet — initialize and train one on the Training tab first.
      </div>
    {:else}
      <div class="grid gap-6" style="grid-template-columns: auto 1fr">
        <!-- Board + transport -->
        <div class="flex flex-col gap-3 items-center">
          <GameBoard {frame} size={400} />
          <div class="flex items-center gap-3 w-full">
            {#if playing}
              <button type="button" class="btn-outline text-sm" onclick={pause}>Pause</button>
            {:else}
              <button type="button" class="btn-outline text-sm" onclick={play} disabled={!result}
                >Play</button
              >
            {/if}
            <input
              type="range"
              min="1"
              max="60"
              step="1"
              bind:value={speed}
              class="flex-1"
              title="Speed"
            />
            <span class="text-xs text-[var(--color-muted)] font-mono w-14 text-right">{speed} fps</span>
          </div>
          {#if result}
            <input
              type="range"
              min="0"
              max={result.frames.length - 1}
              step="1"
              bind:value={idx}
              oninput={pause}
              class="w-full"
              title="Scrub"
            />
            <div class="text-xs text-[var(--color-muted)] font-mono">
              frame {idx + 1}/{result.frames.length} · score {frame?.score ?? 0}
            </div>
          {/if}
        </div>

        <!-- Thinking overlay -->
        <div class="flex flex-col gap-3 min-w-0">
          <div class="card p-4">
            <div class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide mb-2">
              {kind === 'prob'
                ? 'Action probabilities'
                : kind === 'q'
                  ? 'Q-values'
                  : 'Agent decision'}
            </div>
            {#if step?.scores}
              <div class="flex flex-col gap-2">
                {#each step.scores.values as v, ai}
                  <div class="flex items-center gap-2">
                    <span
                      class="w-16 text-xs font-mono"
                      class:text-[var(--color-accent)]={ai === step.action}
                      class:text-[var(--color-muted)]={ai !== step.action}>{ACTION_NAMES[ai]}</span
                    >
                    <div class="flex-1 h-3 rounded bg-[var(--color-border)]/30 overflow-hidden">
                      <div
                        class="h-full rounded"
                        style="width:{(Math.abs(v) / maxScore) * 100}%;
                               background:{ai === step.action
                          ? 'var(--color-accent)'
                          : 'var(--color-muted)'}"
                      ></div>
                    </div>
                    <span class="w-14 text-right text-xs font-mono text-[var(--color-text)]"
                      >{v.toFixed(3)}</span
                    >
                  </div>
                {/each}
              </div>
              <p class="text-xs text-[var(--color-muted)] mt-3">
                The agent chose <span class="text-[var(--color-accent)] font-semibold"
                  >{ACTION_NAMES[step.action]}</span
                >
                {#if step.event === 'eat'}· ate food! 🍎{/if}
                {#if step.event === 'death'}· died 💀{/if}
              </p>
            {:else}
              <p class="text-sm text-[var(--color-muted)]">
                {result
                  ? 'Press Play, or scrub forward to see the agent decide.'
                  : 'Start a game to see the agent think.'}
              </p>
            {/if}
          </div>

          {#if result}
            <div class="card p-4 text-sm text-[var(--color-text)]">
              <div class="flex justify-between">
                <span class="text-[var(--color-muted)]">Final score</span>
                <span class="font-mono font-bold">{result.score}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-[var(--color-muted)]">Game length</span>
                <span class="font-mono">{result.length} steps</span>
              </div>
            </div>
          {/if}
          {#if error}
            <div class="text-xs text-[var(--color-danger)]">{error}</div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

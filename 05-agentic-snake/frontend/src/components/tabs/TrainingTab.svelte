<script lang="ts">
  import {
    algorithm,
    algorithmInfo,
    training,
    applySessionState,
    pushEpisode,
    persistCheckpointPrefs,
  } from '../../state.svelte';
  import { api } from '../../api';
  import ScoreChart from '../ScoreChart.svelte';

  // Chunk size for the train loop — small enough that the chart updates
  // smoothly and Stop is responsive, large enough to amortize HTTP overhead.
  const CHUNK = 5;

  let scores = $derived(training.scoreHistory.map((r) => r.score));
  let lengths = $derived(training.scoreHistory.map((r) => r.length));
  let lastRecord = $derived(
    training.scoreHistory.length ? training.scoreHistory[training.scoreHistory.length - 1] : null
  );

  async function initAgent() {
    training.busy = true;
    training.statusMsg = 'initializing agent…';
    try {
      const summary = await api.initTraining();
      applySessionState({ ...summary, score_history: [] });
      training.lastEval = null;
      training.statusMsg = 'agent ready';
    } catch (e) {
      training.statusMsg = `init failed: ${e}`;
    } finally {
      training.busy = false;
    }
  }

  // Shared driver for "train N" and "train continuously". target=null runs
  // until Stop. The loop is a closure over module-level store state, so it
  // survives tab unmount/remount (run-state lives on the `training` store).
  async function runTraining(target: number | null) {
    if (!training.hasSession || training.running) return;
    training.running = true;
    training.abort = false;
    training.busy = true;
    let done = 0;
    try {
      while (!training.abort && (target === null || done < target)) {
        const n = target === null ? CHUNK : Math.min(CHUNK, target - done);
        const { records } = await api.trainEpisodes(n, { lr: algorithm.hyperparameters.lr });
        for (const r of records) pushEpisode(r);
        done += records.length;
        training.statusMsg =
          target === null
            ? `training continuously — ${training.episode} episodes`
            : `training… ${done}/${target}`;
        if (records.length === 0) break;
      }
      training.statusMsg = training.abort
        ? `stopped at ${training.episode} episodes`
        : `done — ${training.episode} episodes, best ${training.bestScore}`;
    } catch (e) {
      training.statusMsg = `training error: ${e}`;
    } finally {
      training.running = false;
      training.busy = false;
      training.abort = false;
      if (training.autoSave) saveCheckpoint().catch(() => {});
    }
  }

  function stop() {
    training.abort = true;
  }

  async function evaluate() {
    if (!training.hasSession) return;
    training.busy = true;
    training.statusMsg = 'evaluating (greedy)…';
    try {
      const r = await api.evaluate(20);
      training.lastEval = {
        mean_score: r.mean_score,
        best_score: r.best_score,
        mean_length: r.mean_length,
      };
      training.statusMsg = `greedy eval: ${r.mean_score.toFixed(2)} avg · best ${r.best_score}`;
    } catch (e) {
      training.statusMsg = `eval failed: ${e}`;
    } finally {
      training.busy = false;
    }
  }

  // ── Checkpoints ──────────────────────────────────────────────────────
  async function refreshCheckpoints() {
    try {
      training.availableCheckpoints = (await api.listCheckpoints()).files;
    } catch {
      /* ignore */
    }
  }
  $effect(() => {
    refreshCheckpoints();
  });

  async function saveCheckpoint() {
    if (!training.hasSession) return;
    await api.saveCheckpoint(training.checkpointFilename.trim() || 'snake.pt');
    await refreshCheckpoints();
    training.statusMsg = `saved ${training.checkpointFilename}`;
  }

  async function loadCheckpoint(name: string) {
    training.busy = true;
    try {
      applySessionState(await api.loadCheckpoint(name));
      training.checkpointFilename = name;
      persistCheckpointPrefs();
      training.statusMsg = `loaded ${name}`;
    } catch (e) {
      training.statusMsg = `load failed: ${e}`;
    } finally {
      training.busy = false;
    }
  }

  async function deleteCheckpoint(name: string) {
    await api.deleteCheckpoint(name);
    await refreshCheckpoints();
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-6xl mx-auto grid gap-6" style="grid-template-columns: 320px 1fr">
    <!-- Left: controls -->
    <div class="flex flex-col gap-4">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Training</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          {algorithmInfo()?.label ?? 'Pick an algorithm'} · {training.hasSession
            ? `${training.episode} episodes trained`
            : 'no agent yet'}
        </p>
      </div>

      <button
        type="button"
        class="btn-capsule"
        disabled={training.running || training.busy}
        onclick={initAgent}
      >
        {training.hasSession ? 'Re-initialize Agent' : 'Initialize Agent'}
      </button>

      <div class="card p-4 flex flex-col gap-3">
        <label class="flex items-center justify-between gap-3 text-sm">
          <span class="text-[var(--color-text)]">Episodes</span>
          <input
            type="number"
            min="1"
            class="input w-24 text-right font-mono"
            bind:value={training.episodesPerRun}
          />
        </label>
        <div class="flex gap-2">
          <button
            type="button"
            class="btn-primary flex-1"
            disabled={!training.hasSession || training.running || training.busy}
            onclick={() => runTraining(training.episodesPerRun)}
          >
            Train {training.episodesPerRun}
          </button>
          {#if training.running}
            <button type="button" class="btn-danger flex-1" onclick={stop}>Stop</button>
          {:else}
            <button
              type="button"
              class="btn-outline flex-1"
              disabled={!training.hasSession || training.busy}
              onclick={() => runTraining(null)}
            >
              Continuous
            </button>
          {/if}
        </div>
        <button
          type="button"
          class="btn-ghost text-sm"
          disabled={!training.hasSession || training.running || training.busy}
          onclick={evaluate}
        >
          Evaluate (greedy, 20 games)
        </button>
      </div>

      <!-- Checkpoints -->
      <div class="card p-4 flex flex-col gap-2">
        <span class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide"
          >Checkpoints</span
        >
        <div class="flex gap-2">
          <input
            class="input flex-1 font-mono text-sm"
            bind:value={training.checkpointFilename}
            oninput={persistCheckpointPrefs}
          />
          <button
            type="button"
            class="btn-outline text-sm"
            disabled={!training.hasSession}
            onclick={saveCheckpoint}>Save</button
          >
        </div>
        <label class="flex items-center gap-2 text-xs text-[var(--color-text)]">
          <input type="checkbox" bind:checked={training.autoSave} onchange={persistCheckpointPrefs} />
          Auto-save after each run
        </label>
        <label class="flex items-center gap-2 text-xs text-[var(--color-text)]">
          <input
            type="checkbox"
            bind:checked={training.autoLoadOnRestart}
            onchange={persistCheckpointPrefs}
          />
          Auto-load on restart
        </label>
        {#if training.availableCheckpoints.length}
          <div class="flex flex-col gap-1 mt-1">
            {#each training.availableCheckpoints as f}
              <div class="flex items-center gap-2 text-xs">
                <button
                  type="button"
                  class="flex-1 text-left truncate hover:text-[var(--color-accent)] font-mono"
                  onclick={() => loadCheckpoint(f.name)}
                  title="Load {f.name}">{f.name}</button
                >
                <button
                  type="button"
                  class="text-[var(--color-muted)] hover:text-[var(--color-danger)]"
                  onclick={() => deleteCheckpoint(f.name)}
                  title="Delete">✕</button
                >
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Right: live metrics -->
    <div class="flex flex-col gap-4 min-w-0">
      <div class="grid gap-3" style="grid-template-columns: repeat(4, 1fr)">
        {#each [{ label: 'Episodes', val: String(training.episode) }, { label: 'Best score', val: String(training.bestScore) }, { label: 'Last score', val: lastRecord ? String(lastRecord.score) : '—' }, { label: 'Params', val: training.paramCount > 0 ? training.paramCount.toLocaleString() : 'tabular' }] as stat}
          <div class="card p-3">
            <div class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">{stat.label}</div>
            <div class="text-lg font-bold text-[var(--color-heading)] font-mono">{stat.val}</div>
          </div>
        {/each}
      </div>

      <ScoreChart values={scores} label="Score per episode (apples eaten)" height={240} />
      <ScoreChart
        values={lengths}
        label="Episode length (steps survived)"
        height={180}
        color="var(--color-muted)"
      />

      {#if lastRecord}
        <div class="card p-3 text-xs text-[var(--color-muted)] flex flex-wrap gap-x-5 gap-y-1 font-mono">
          {#if lastRecord.epsilon !== undefined}<span>ε = {lastRecord.epsilon.toFixed(3)}</span>{/if}
          {#if lastRecord.loss !== undefined}<span>loss = {lastRecord.loss.toFixed(4)}</span>{/if}
          {#if lastRecord.q_states !== undefined}<span>Q-states = {lastRecord.q_states}</span>{/if}
          {#if lastRecord.buffer !== undefined}<span>buffer = {lastRecord.buffer.toLocaleString()}</span>{/if}
        </div>
      {/if}

      <div class="text-xs text-[var(--color-muted)] font-mono">{training.statusMsg}</div>
    </div>
  </div>
</div>

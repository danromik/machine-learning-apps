<script lang="ts">
  import {
    algorithm,
    algorithmInfo,
    cube,
    training,
    ui,
    applySessionState,
    pushIteration,
    persistCheckpointPrefs,
  } from '../../state.svelte';
  import { api, type SolveResult } from '../../api';
  import ScoreChart from '../ScoreChart.svelte';
  import CubePlayer from '../CubePlayer.svelte';

  // Chunk size for the foreground train loop — small enough that the chart
  // updates smoothly and Stop is responsive.
  const CHUNK = 3;

  // Which training interface is shown (foreground hands-on vs background run).
  let trainMode = $state<'foreground' | 'background'>('foreground');

  // The "Play last training round" 3D view's solve (local to this tab).
  let playResult = $state<SolveResult | null>(null);
  let playBusy = $state(false);

  let losses = $derived(training.lossHistory.map((r) => r.loss));
  let lastRecord = $derived(
    training.lossHistory.length ? training.lossHistory[training.lossHistory.length - 1] : null
  );
  let runActive = $derived(!!training.run?.running);
  let solveRows = $derived(
    Object.entries(training.solveRateByK)
      .map(([k, v]) => ({ k: Number(k), rate: v }))
      .sort((a, b) => a.k - b.k)
  );

  async function initAgent() {
    training.busy = true;
    training.statusMsg = 'initializing agent…';
    try {
      const summary = await api.initTraining();
      applySessionState({ ...summary, loss_history: [] });
      training.lastEval = null;
      playResult = null;
      training.statusMsg = 'agent ready';
    } catch (e) {
      training.statusMsg = `init failed: ${e}`;
    } finally {
      training.busy = false;
    }
  }

  // Foreground driver. target=null runs until Stop. Closure over module-level
  // store state so it survives tab unmount/remount. When auto-advance is on, it
  // evaluates every `evalEveryN` iterations and promotes the curriculum depth k
  // once the current depth clears `promoteAt` — the same rule the background run
  // uses, but driven here so foreground training ramps too.
  async function runTraining(target: number | null) {
    if (!training.hasSession || training.running || runActive) return;
    training.running = true;
    training.abort = false;
    training.busy = true;
    let done = 0;
    let sinceEval = 0;
    try {
      while (!training.abort && (target === null || done < target)) {
        const n = target === null ? CHUNK : Math.min(CHUNK, target - done);
        const { records } = await api.trainIterations(n, training.currentK, {
          lr: algorithm.hyperparameters.lr,
        });
        for (const r of records) pushIteration(r);
        done += records.length;
        sinceEval += records.length;
        if (training.autoAdvance && sinceEval >= training.evalEveryN) {
          sinceEval = 0;
          const promoted = await maybeAutoAdvance();
          if (promoted) continue; // status already set by the promotion
        }
        training.statusMsg =
          target === null
            ? `training continuously — ${training.iteration} iters · k=${training.currentK}`
            : `training… ${done}/${target} · k=${training.currentK}`;
        if (records.length === 0) break;
      }
      training.statusMsg = training.abort
        ? `stopped at ${training.iteration} iterations`
        : `done — ${training.iteration} iterations · k=${training.currentK}`;
    } catch (e) {
      training.statusMsg = `training error: ${e}`;
    } finally {
      training.running = false;
      training.busy = false;
      training.abort = false;
    }
  }

  // Evaluate solve-rate at the current depth; promote k if it clears the bar.
  // Returns true if it promoted.
  async function maybeAutoAdvance(): Promise<boolean> {
    try {
      const r = await api.evaluate(training.evalN, training.currentK);
      training.lastEval = {
        solve_rate: r.solve_rate,
        mean_solution_len: r.mean_solution_len,
        k: r.k,
      };
      training.solveRateByK = { ...training.solveRateByK, [String(r.k)]: r.solve_rate };
      if (r.solve_rate >= cube.curriculum.promoteAt && training.currentK < cube.curriculum.maxK) {
        training.currentK += 1;
        training.statusMsg =
          `mastered k=${r.k} (${(r.solve_rate * 100).toFixed(0)}%) — advanced to k=${training.currentK}`;
        return true;
      }
    } catch {
      /* eval failed — keep training at the current depth */
    }
    return false;
  }

  function stop() {
    training.abort = true;
  }

  function setK(k: number) {
    training.currentK = Math.max(1, Math.min(cube.curriculum.maxK, k));
  }

  async function playLastRound() {
    if (!training.hasSession || playBusy) return;
    playBusy = true;
    training.statusMsg = `solving a k=${training.currentK} scramble with the current model…`;
    try {
      playResult = await api.play(training.currentK);
      training.statusMsg = playResult.solved
        ? `solved a k=${training.currentK} scramble in ${playResult.solution_len} moves`
        : `didn't solve a k=${training.currentK} scramble (${playResult.solution_len} moves tried)`;
    } catch (e) {
      training.statusMsg = `play failed: ${e}`;
    } finally {
      playBusy = false;
    }
  }

  // ── Background run ───────────────────────────────────────────────────────
  async function startRun() {
    try {
      training.run = await api.runStart({
        start_k: cube.curriculum.startK,
        max_k: cube.curriculum.maxK,
        promote_at: cube.curriculum.promoteAt,
        eval_every: training.evalEveryN,
        cadence_minutes: training.cadenceMinutes,
      });
      training.statusMsg = 'background run started';
    } catch (e) {
      training.statusMsg = `couldn't start run: ${e}`;
    }
  }

  async function stopRun() {
    try {
      training.run = await api.runStop();
      training.statusMsg = 'background run stopped';
    } catch (e) {
      training.statusMsg = `stop failed: ${e}`;
    }
  }

  // ── Checkpoints ──────────────────────────────────────────────────────────
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
    await api.saveCheckpoint(training.checkpointFilename.trim() || 'cube.pt');
    await refreshCheckpoints();
    training.statusMsg = `saved ${training.checkpointFilename}`;
  }

  async function loadCheckpoint(name: string) {
    if (runActive) return;
    training.busy = true;
    try {
      applySessionState(await api.loadCheckpoint(name));
      training.checkpointFilename = name;
      persistCheckpointPrefs();
      playResult = null;
      training.statusMsg = `loaded ${name}`;
    } catch (e) {
      training.statusMsg = `load failed: ${e}`;
    } finally {
      training.busy = false;
    }
  }

  async function deleteCheckpoint(name: string) {
    // Destructive — confirm first. Bundled checkpoints are guarded server-side
    // too and shouldn't reach here (their button is disabled).
    if (typeof window !== 'undefined' && !window.confirm(`Delete checkpoint "${name}"? This can't be undone.`)) {
      return;
    }
    try {
      await api.deleteCheckpoint(name);
      await refreshCheckpoints();
      training.statusMsg = `deleted ${name}`;
    } catch (e) {
      training.statusMsg = `delete failed: ${e}`;
    }
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-6xl mx-auto grid gap-6" style="grid-template-columns: 340px 1fr">
    <!-- Left: controls -->
    <div class="flex flex-col gap-4">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Training</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          {algorithmInfo()?.label ?? 'value iteration'} · {training.hasSession
            ? `${training.iteration} iters · k=${training.currentK}`
            : 'no agent yet'}
        </p>
      </div>

      <button
        type="button"
        class="btn-capsule"
        disabled={training.running || training.busy || runActive}
        onclick={initAgent}
      >
        {training.hasSession ? 'Re-initialize Agent' : 'Initialize Agent'}
      </button>

      <!-- Training mode: Foreground / Background sub-tabs -->
      <div class="card p-4 flex flex-col gap-3">
        <div class="flex gap-1 p-0.5 rounded-md bg-[var(--color-border)]/30">
          <button
            type="button"
            class="subtab"
            class:subtab-active={trainMode === 'foreground'}
            onclick={() => (trainMode = 'foreground')}>Foreground</button
          >
          <button
            type="button"
            class="subtab"
            class:subtab-active={trainMode === 'background'}
            onclick={() => (trainMode = 'background')}>Background</button
          >
        </div>

        {#if trainMode === 'foreground'}
          <p class="text-[11px] text-[var(--color-muted)]">
            Hands-on training at a fixed (or auto-advancing) curriculum depth.
          </p>

          <!-- Curriculum depth control -->
          <div class="flex items-center justify-between gap-3 text-sm">
            <span class="text-[var(--color-text)]">Scramble depth k</span>
            <div class="flex items-center gap-1">
              <button
                type="button"
                class="btn-outline px-2 py-0.5 text-sm"
                disabled={runActive || training.currentK <= 1}
                onclick={() => setK(training.currentK - 1)}
                title="Train on shallower scrambles">−</button
              >
              <span class="w-8 text-center font-mono text-[var(--color-heading)]"
                >{training.currentK}</span
              >
              <button
                type="button"
                class="btn-outline px-2 py-0.5 text-sm"
                disabled={runActive || training.currentK >= cube.curriculum.maxK}
                onclick={() => setK(training.currentK + 1)}
                title="Train on deeper scrambles">+</button
              >
            </div>
          </div>
          <label class="flex items-center gap-2 text-xs text-[var(--color-text)]">
            <input type="checkbox" bind:checked={training.autoAdvance} disabled={runActive} />
            Auto-advance k when solved
            <span class="text-[var(--color-muted)]"
              >(≥{(cube.curriculum.promoteAt * 100).toFixed(0)}% every {training.evalEveryN})</span
            >
          </label>

          <label class="flex items-center justify-between gap-3 text-sm">
            <span class="text-[var(--color-text)]">Iterations</span>
            <input
              type="number"
              min="1"
              class="input w-24 text-right font-mono"
              bind:value={training.iterationsPerRun}
            />
          </label>
          <div class="flex gap-2">
            <button
              type="button"
              class="btn-primary flex-1"
              disabled={!training.hasSession || training.running || training.busy || runActive}
              onclick={() => runTraining(training.iterationsPerRun)}
            >
              Train {training.iterationsPerRun}
            </button>
            {#if training.running}
              <button type="button" class="btn-danger flex-1" onclick={stop}>Stop</button>
            {:else}
              <button
                type="button"
                class="btn-outline flex-1"
                disabled={!training.hasSession || training.busy || runActive}
                onclick={() => runTraining(null)}
              >
                Continuous
              </button>
            {/if}
          </div>
          {#if runActive}
            <p class="text-[11px] text-[var(--color-muted)]">
              Foreground controls are disabled while the background run owns the agent.
            </p>
          {/if}
        {:else}
          <p class="text-[11px] text-[var(--color-muted)]">
            Trains on its own thread, ramps the curriculum, and checkpoints itself.
            The RL Coach checks in periodically and writes the Progress Report.
          </p>
          <label class="flex items-center justify-between gap-3 text-sm">
            <span class="text-[var(--color-text)]">Check-in every (min)</span>
            <input
              type="number"
              min="1"
              class="input w-20 text-right font-mono"
              bind:value={training.cadenceMinutes}
            />
          </label>
          {#if runActive}
            <button type="button" class="btn-danger" onclick={stopRun}>Stop Run</button>
          {:else}
            <button type="button" class="btn-primary" onclick={startRun}>Start Overnight Run</button>
          {/if}
          {#if training.run && training.run.state !== 'idle'}
            <div class="text-[11px] font-mono text-[var(--color-muted)]">
              {training.run.state} · iter {training.run.iteration} · k={training.run.current_k}
              {#if training.run.last_checkpoint}· ✓ {training.run.last_checkpoint}{/if}
            </div>
          {/if}
        {/if}
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
                {#if f.protected}
                  <span
                    class="text-[var(--color-muted)]/50 cursor-not-allowed"
                    title="Bundled checkpoint — can't be deleted">🔒</span
                  >
                {:else}
                  <button
                    type="button"
                    class="text-[var(--color-muted)] hover:text-[var(--color-danger)]"
                    onclick={() => deleteCheckpoint(f.name)}
                    title="Delete">✕</button
                  >
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Right: live metrics -->
    <div class="flex flex-col gap-4 min-w-0">
      <div class="grid gap-3" style="grid-template-columns: repeat(4, 1fr)">
        {#each [{ label: 'Iterations', val: String(training.iteration) }, { label: 'Curriculum k', val: String(training.currentK) }, { label: 'Last loss', val: lastRecord ? lastRecord.loss.toFixed(4) : '—' }, { label: 'Params', val: training.paramCount > 0 ? training.paramCount.toLocaleString() : '—' }] as stat}
          <div class="card p-3">
            <div class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
              {stat.label}
            </div>
            <div class="text-lg font-bold text-[var(--color-heading)] font-mono">{stat.val}</div>
          </div>
        {/each}
      </div>

      <ScoreChart values={losses} label="Training loss (per iteration)" height={240} />

      <!-- Solve-rate + last-round 3D view, side by side -->
      <div class="grid gap-4" style="grid-template-columns: 1fr 1fr">
        <!-- Solve-rate by curriculum depth -->
        <div class="card p-4">
          <div class="text-xs font-semibold text-[var(--color-heading)] mb-2">
            Solve-rate by scramble depth
          </div>
          {#if solveRows.length === 0}
            <div class="text-sm text-[var(--color-muted)]">
              Turn on auto-advance, or start a background run, to measure solve-rate.
            </div>
          {:else}
            <div class="flex flex-col gap-1.5">
              {#each solveRows as row}
                <div class="flex items-center gap-2 text-xs">
                  <span class="w-10 font-mono text-[var(--color-muted)]">k={row.k}</span>
                  <div class="flex-1 h-3 rounded bg-[var(--color-border)]/40 overflow-hidden">
                    <div
                      class="h-full rounded"
                      style="width: {(row.rate * 100).toFixed(0)}%; background: var(--color-accent)"
                    ></div>
                  </div>
                  <span class="w-12 text-right font-mono text-[var(--color-text)]"
                    >{(row.rate * 100).toFixed(0)}%</span
                  >
                </div>
              {/each}
            </div>
          {/if}
        </div>

        <!-- Last training round (3D) -->
        <div class="card p-3 flex flex-col gap-2">
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs font-semibold text-[var(--color-heading)]">Last training round</span>
            <button
              type="button"
              class="btn-outline text-xs"
              disabled={!training.hasSession || playBusy}
              onclick={playLastRound}
              title="Solve a k={training.currentK} scramble with the current model"
            >
              {playBusy ? 'Solving…' : 'Play last training round'}
            </button>
          </div>
          <CubePlayer result={playResult} size={cube.size} height="240px" />
          {#if !playResult}
            <p class="text-[11px] text-[var(--color-muted)]">
              Shows the current model attempting a depth-{training.currentK} scramble.
            </p>
          {/if}
        </div>
      </div>

      <div class="text-xs text-[var(--color-muted)] font-mono">{training.statusMsg}</div>
      <div>
        <button type="button" class="btn-outline text-sm" onclick={() => (ui.activeTab = 'watch')}>
          Watch it solve →
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .subtab {
    flex: 1;
    padding: 0.25rem 0.5rem;
    border-radius: 0.3rem;
    font-size: 0.8rem;
    color: var(--color-muted);
    transition: all 0.12s;
  }
  .subtab-active {
    background: var(--color-surface);
    color: var(--color-heading);
    box-shadow: 0 1px 2px rgb(0 0 0 / 0.1);
  }
</style>

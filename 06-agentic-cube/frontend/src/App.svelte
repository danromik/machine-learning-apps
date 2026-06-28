<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import ChatPane from './components/chat/ChatPane.svelte';
  import { applyChatEvent } from './components/chat/chatReducer';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import CubeTab from './components/tabs/CubeTab.svelte';
  import AlgorithmTab from './components/tabs/AlgorithmTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import WatchTab from './components/tabs/WatchTab.svelte';
  import ProgressReportTab from './components/tabs/ProgressReportTab.svelte';
  import DebriefTab from './components/tabs/DebriefTab.svelte';
  import {
    ui,
    cube,
    algorithm,
    training,
    report,
    chat,
    algorithmInfo,
    applySessionState,
    pushIteration,
    persistUiPrefs,
    type TabId,
  } from './state.svelte';
  import { api, openStateSocket, type StateBroadcastEvent, type AlgorithmId } from './api';

  const TABS: { id: TabId; label: string }[] = [
    { id: 'orientation', label: 'Orientation' },
    { id: 'cube', label: 'Cube' },
    { id: 'algorithm', label: 'Algorithm' },
    { id: 'training', label: 'Training' },
    { id: 'watch', label: 'Watch' },
    { id: 'progress-report', label: 'Progress Report' },
    { id: 'debrief', label: 'Debrief' },
  ];

  let tabSubtitles = $derived.by(() => {
    const info = algorithmInfo();
    const subs: Record<TabId, string> = {
      orientation: '',
      cube: `${cube.size}×${cube.size}×${cube.size} · k≤${cube.curriculum.maxK}`,
      algorithm: info ? info.label : '',
      training: training.hasSession
        ? `${training.iteration} iters · k=${training.currentK}`
        : 'no agent',
      watch: '',
      'progress-report': training.run && training.run.state !== 'idle'
        ? `run ${training.run.state}`
        : report.markdown ? 'report ready' : 'no run yet',
      debrief: report.final ? 'final report' : training.hasSession ? `k=${training.currentK}` : '—',
    };
    return subs;
  });

  // ── Catalog preload ───────────────────────────────────────────────────
  $effect(() => {
    if (algorithm.loaded) return;
    (async () => {
      try {
        const { algorithms } = await api.catalog();
        algorithm.catalog = algorithms;
        if (!algorithm.hyperparameters || Object.keys(algorithm.hyperparameters).length === 0) {
          const cur = algorithms.find((a) => a.id === algorithm.algo);
          if (cur) algorithm.hyperparameters = { ...cur.default_hyperparameters };
        }
        algorithm.loaded = true;
      } catch (e) {
        console.error('catalog preload failed', e);
      }
    })();
  });

  // Persist active tab.
  $effect(() => {
    persistUiPrefs();
  });

  // ── Pipeline-state sync (UI ↔ backend mirror ↔ agent) ─────────────────
  let lastSyncedSig: string | null = null;
  let stateLoaded = $state(false);
  let suppressLocalSync = false;

  function _localStateForSync() {
    return {
      environment: {
        size: cube.size,
        curriculum: { ...cube.curriculum },
      },
      algorithm: {
        algo: algorithm.algo,
        hyperparameters: { ...algorithm.hyperparameters },
      },
      training: {
        iterationsPerRun: training.iterationsPerRun,
        evalEveryN: training.evalEveryN,
        evalN: training.evalN,
        cadenceMinutes: training.cadenceMinutes,
      },
    };
  }

  function _applyServerState(patch: Record<string, any>) {
    suppressLocalSync = true;
    try {
      if (patch.environment) {
        const e = patch.environment;
        if (e.size === 2 || e.size === 3) cube.size = e.size;
        if (e.curriculum) cube.curriculum = { ...cube.curriculum, ...e.curriculum };
      }
      if (patch.algorithm) {
        const a = patch.algorithm;
        if (a.algo) algorithm.algo = a.algo as AlgorithmId;
        if (a.hyperparameters) {
          algorithm.hyperparameters = { ...algorithm.hyperparameters, ...a.hyperparameters };
        }
      }
      if (patch.training) {
        const t = patch.training;
        if (typeof t.iterationsPerRun === 'number') training.iterationsPerRun = t.iterationsPerRun;
        if (typeof t.evalEveryN === 'number') training.evalEveryN = t.evalEveryN;
        if (typeof t.evalN === 'number') training.evalN = t.evalN;
        if (typeof t.cadenceMinutes === 'number') training.cadenceMinutes = t.cadenceMinutes;
      }
      lastSyncedSig = JSON.stringify(_localStateForSync());
    } finally {
      queueMicrotask(() => {
        suppressLocalSync = false;
      });
    }
  }

  // Initial state load: backend is authoritative on boot.
  let initialStateFetched = false;
  $effect(() => {
    if (!algorithm.loaded || initialStateFetched) return;
    initialStateFetched = true;
    (async () => {
      try {
        const snap = await api.getState();
        _applyServerState(snap as unknown as Record<string, any>);
        try {
          applySessionState(await api.trainingState());
        } catch {
          // no session — fine
        }
        try {
          training.run = await api.runStatus();
        } catch { /* ignore */ }
        try {
          const r = await api.report();
          report.markdown = r.markdown;
          report.final = r.final;
          report.updatedAt = r.updated_at;
        } catch { /* ignore */ }
        stateLoaded = true;
      } catch (e) {
        console.warn('initial state fetch failed', e);
        stateLoaded = true;
      }
    })();
  });

  // Auto-sync local edits → backend, debounced.
  let syncTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const sig = JSON.stringify(_localStateForSync());
    if (!stateLoaded || suppressLocalSync) return;
    if (lastSyncedSig === null) {
      lastSyncedSig = sig;
      return;
    }
    if (sig === lastSyncedSig) return;
    if (syncTimer) clearTimeout(syncTimer);
    syncTimer = setTimeout(() => {
      lastSyncedSig = sig;
      api
        .patchState(_localStateForSync() as unknown as Record<string, unknown>)
        .catch((e) => console.warn('state patch push failed', e));
    }, 200);
  });

  async function refreshRunStatus() {
    try {
      training.run = await api.runStatus();
    } catch { /* ignore */ }
  }

  // ── WebSocket: state patches + live training/report/agent events ──────
  let stateWs: WebSocket | null = null;
  function connectStateWs() {
    if (typeof window === 'undefined') return;
    stateWs = openStateSocket(
      (ev: StateBroadcastEvent) => {
        if (ev.type === 'state_replace') {
          _applyServerState(ev.state as unknown as Record<string, any>);
        } else if (ev.type === 'state_patch') {
          _applyServerState(ev.patch);
        } else if (ev.type === 'training_session') {
          if (ev.hasSession && ev.summary) {
            applySessionState({ ...ev.summary, loss_history: ev.lossHistory });
          } else if (!ev.hasSession) {
            applySessionState({ has_session: false });
          }
        } else if (ev.type === 'trainer_progress') {
          training.hasSession = true;
          pushIteration(ev.record);
          training.currentK = ev.current_k;
          training.solveRateByK = ev.solve_rate_by_k;
          if (training.run) {
            training.run = { ...training.run, iteration: ev.iteration, current_k: ev.current_k };
          }
        } else if (ev.type === 'trainer_status') {
          if (ev.eval) {
            training.lastEval = {
              solve_rate: ev.eval.solve_rate,
              mean_solution_len: ev.eval.mean_solution_len,
              k: ev.eval.k,
            };
            training.solveRateByK = { ...training.solveRateByK, [String(ev.eval.k)]: ev.eval.solve_rate };
          }
          if (typeof ev.current_k === 'number') training.currentK = ev.current_k;
          // A status transition is rare — refetch the full snapshot.
          refreshRunStatus();
        } else if (ev.type === 'report_update') {
          report.markdown = ev.markdown;
          report.final = false;
          report.updatedAt = ev.updated_at;
        } else if (ev.type === 'report_final') {
          report.markdown = ev.markdown;
          report.final = true;
          report.updatedAt = ev.updated_at;
          ui.activeTab = 'debrief';
        } else if (ev.type === 'agent_event') {
          applyChatEvent(ev.event);
        }
      },
      () => setTimeout(connectStateWs, 1500)
    );
  }
  $effect(() => {
    connectStateWs();
    return () => {
      if (stateWs && stateWs.readyState <= 1) stateWs.close();
    };
  });

  // Light polling fallback while a run is active (covers any dropped WS events).
  $effect(() => {
    if (!training.run?.running) return;
    const id = setInterval(refreshRunStatus, 5000);
    return () => clearInterval(id);
  });
</script>

<div class="h-screen w-screen flex flex-col overflow-hidden">
  <Header />

  <nav
    class="flex items-center gap-1.5 px-4 py-1.5 border-b border-[var(--color-border)]
           bg-[var(--color-surface)]/40 shrink-0 overflow-x-auto"
  >
    {#each TABS as tab, i}
      <button
        type="button"
        class="tab-button"
        class:tab-button-active={ui.activeTab === tab.id}
        onclick={() => (ui.activeTab = tab.id)}
      >
        <span class="tab-number">{i}</span>
        <span class="tab-label-stack">
          <span class="tab-label-text">{tab.label}</span>
          <span class="tab-subtitle">{tabSubtitles[tab.id] || ' '}</span>
        </span>
      </button>
    {/each}
  </nav>

  <div class="flex-1 min-h-0 flex overflow-hidden">
    <main class="flex-1 min-w-0 min-h-0 overflow-auto">
      {#if ui.activeTab === 'orientation'}
        <OrientationTab />
      {:else if ui.activeTab === 'cube'}
        <CubeTab />
      {:else if ui.activeTab === 'algorithm'}
        <AlgorithmTab />
      {:else if ui.activeTab === 'training'}
        <TrainingTab />
      {:else if ui.activeTab === 'watch'}
        <WatchTab />
      {:else if ui.activeTab === 'progress-report'}
        <ProgressReportTab />
      {:else if ui.activeTab === 'debrief'}
        <DebriefTab />
      {/if}
    </main>

    {#if chat.visible}
      <ChatPane />
    {/if}
  </div>

  <StatusBar />
</div>

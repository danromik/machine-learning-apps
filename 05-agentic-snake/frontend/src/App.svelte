<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import ChatPane from './components/chat/ChatPane.svelte';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import EnvironmentTab from './components/tabs/EnvironmentTab.svelte';
  import AlgorithmTab from './components/tabs/AlgorithmTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import WatchTab from './components/tabs/WatchTab.svelte';
  import DebriefTab from './components/tabs/DebriefTab.svelte';
  import {
    ui,
    environment,
    algorithm,
    training,
    chat,
    algorithmInfo,
    applySessionState,
    pushEpisode,
    persistUiPrefs,
    type TabId,
  } from './state.svelte';
  import { api, openStateSocket, type StateBroadcastEvent, type AlgorithmId } from './api';

  const TABS: { id: TabId; label: string }[] = [
    { id: 'orientation', label: 'Orientation' },
    { id: 'environment', label: 'Environment' },
    { id: 'algorithm', label: 'Algorithm' },
    { id: 'training', label: 'Training' },
    { id: 'watch', label: 'Watch' },
    { id: 'debrief', label: 'Debrief' },
  ];

  let tabSubtitles = $derived.by(() => {
    const info = algorithmInfo();
    const subs: Record<TabId, string> = {
      orientation: '',
      environment: `${environment.width}×${environment.height} · ${
        environment.observation === 'grid' ? 'grid obs' : '11 features'
      }`,
      algorithm: info ? info.label : '',
      training: training.hasSession
        ? `${training.episode} episodes · best ${training.bestScore}`
        : 'no agent',
      watch: '',
      debrief: training.hasSession
        ? training.lastEval
          ? `eval ${training.lastEval.mean_score.toFixed(1)} avg`
          : `${training.episode} episodes`
        : 'no agent',
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

  // Auto-load on restart.
  let autoLoadAttempted = false;
  $effect(() => {
    if (!algorithm.loaded || autoLoadAttempted) return;
    if (!training.autoLoadOnRestart) return;
    const fname = training.checkpointFilename.trim();
    if (!fname) return;
    autoLoadAttempted = true;
    (async () => {
      try {
        const { files } = await api.listCheckpoints();
        training.availableCheckpoints = files;
        const withExt = fname.endsWith('.pt') ? fname : `${fname}.pt`;
        if (!files.some((c) => c.name === withExt)) return;
        applySessionState(await api.loadCheckpoint(fname));
      } catch (e) {
        console.warn('auto-load failed:', e);
      }
    })();
  });

  // ── Pipeline-state sync (UI ↔ backend mirror ↔ agent) ─────────────────
  let lastSyncedSig: string | null = null;
  let stateLoaded = $state(false);
  let suppressLocalSync = false;

  function _localStateForSync() {
    return {
      environment: {
        width: environment.width,
        height: environment.height,
        observation: environment.observation,
        reward: { ...environment.reward },
      },
      algorithm: {
        algo: algorithm.algo,
        hyperparameters: { ...algorithm.hyperparameters },
      },
      training: {
        episodesPerRun: training.episodesPerRun,
        evalEveryN: training.evalEveryN,
      },
    };
  }

  function _applyServerState(patch: Record<string, any>) {
    suppressLocalSync = true;
    try {
      if (patch.environment) {
        const e = patch.environment;
        if (typeof e.width === 'number') environment.width = e.width;
        if (typeof e.height === 'number') environment.height = e.height;
        if (e.observation === 'features' || e.observation === 'grid')
          environment.observation = e.observation;
        if (e.reward) environment.reward = { ...environment.reward, ...e.reward };
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
        if (typeof t.episodesPerRun === 'number') training.episodesPerRun = t.episodesPerRun;
        if (typeof t.evalEveryN === 'number') training.evalEveryN = t.evalEveryN;
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
        // Also pull any live session that already exists (e.g. agent built one).
        try {
          applySessionState(await api.trainingState());
        } catch {
          // no session — fine
        }
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

  // ── WebSocket: state patches + live training ticks ────────────────────
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
            applySessionState({ ...ev.summary, score_history: ev.scoreHistory });
          } else if (!ev.hasSession) {
            applySessionState({ has_session: false });
          }
        } else if (ev.type === 'episode_tick') {
          training.hasSession = true;
          pushEpisode(ev.record);
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
</script>

<div class="h-screen w-screen flex flex-col overflow-hidden">
  <Header />

  <nav
    class="flex items-center gap-1.5 px-4 py-1.5 border-b border-[var(--color-border)]
           bg-[var(--color-surface)]/40 shrink-0"
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
          <span class="tab-subtitle">{tabSubtitles[tab.id] || ' '}</span>
        </span>
      </button>
    {/each}
  </nav>

  <div class="flex-1 min-h-0 flex overflow-hidden">
    <main class="flex-1 min-w-0 min-h-0 overflow-auto">
      {#if ui.activeTab === 'orientation'}
        <OrientationTab />
      {:else if ui.activeTab === 'environment'}
        <EnvironmentTab />
      {:else if ui.activeTab === 'algorithm'}
        <AlgorithmTab />
      {:else if ui.activeTab === 'training'}
        <TrainingTab />
      {:else if ui.activeTab === 'watch'}
        <WatchTab />
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

<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import ChatPane from './components/chat/ChatPane.svelte';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import DataSynthesisTab from './components/tabs/DataSynthesisTab.svelte';
  import ArchitectureTab from './components/tabs/ArchitectureTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import InferenceTab from './components/tabs/InferenceTab.svelte';
  import DebriefTab from './components/tabs/DebriefTab.svelte';
  import {
    ui,
    synthesis,
    architecture,
    training,
    chat,
    INPUT_SHAPE,
    applyPreset,
    applyCheckpointResponse,
    persistUiPrefs,
    type TabId,
    type Optimizer,
  } from './state.svelte';
  import { api, openStateSocket, type StateBroadcastEvent } from './api';
  import type { Layer, LayerType } from './components/tabs/architecture/computeArchitecture';
  import {
    computeArchitecture,
    formatCount,
  } from './components/tabs/architecture/computeArchitecture';

  const TABS: { id: TabId; label: string }[] = [
    { id: 'orientation', label: 'Orientation' },
    { id: 'data', label: 'Data Synthesis' },
    { id: 'architecture', label: 'Model Architecture' },
    { id: 'training', label: 'Training' },
    { id: 'inference', label: 'Inference' },
    { id: 'debrief', label: 'Debrief' },
  ];

  // Number of symbols across all selected categories — feeds both the Data
  // Synthesis subtitle and the Architecture/Training computations.
  let symbolCount = $derived.by(() => {
    if (!synthesis.loaded) return 0;
    let n = 0;
    for (const c of synthesis.categories) {
      if (synthesis.selectedCategories[c.id]) n += c.count;
    }
    return n;
  });

  let fontCount = $derived.by(() => {
    if (!synthesis.loaded) return 0;
    let n = 0;
    for (const f of synthesis.fonts) {
      const u = synthesis.fontUsage[f.family];
      if (u === 'train' || u === 'val') n++;
    }
    return n;
  });

  let totalParams = $derived.by(() => {
    const numClasses = symbolCount || 10;
    return computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
      .totalParams;
  });

  // Number of batches required for one "epoch" — defined here as enough
  // batches that each symbol gets, on average, samplesPerSymbolPerEpoch
  // training examples. Uses the synthesis-derived class count when there's
  // no live session.
  let batchesPerEpoch = $derived.by(() => {
    const classes = training.numClasses || symbolCount;
    const bs = architecture.hyperparameters.batch_size;
    if (classes <= 0 || bs <= 0) return 0;
    return Math.max(
      1,
      Math.ceil((classes * training.samplesPerSymbolPerEpoch) / bs)
    );
  });

  let tabSubtitles = $derived.by(() => {
    const subs: Record<TabId, string> = {
      orientation: '',
      data: synthesis.loaded
        ? `${symbolCount} symbols · ${fontCount} fonts`
        : '',
      architecture: `${architecture.layers.length} layers · ${formatCount(totalParams)} weights`,
      training: '',
      inference: '',
      debrief: '',
    };

    if (training.hasSession && batchesPerEpoch > 0) {
      const epochs = Math.floor(training.step / batchesPerEpoch);
      const batchInEpoch = training.step % batchesPerEpoch;
      const accStr =
        training.lastAccuracy === null
          ? '— accuracy'
          : `${(training.lastAccuracy * 100).toFixed(1)}% accuracy`;
      subs.training = `${epochs} epochs · ${batchInEpoch} batches · ${accStr}`;
    } else {
      subs.training = 'no session';
    }

    // Debrief subtitle mirrors training's progress signal so the user
    // can see at a glance whether they have something to celebrate.
    if (training.hasSession && training.step > 0) {
      if (training.lastAccuracy !== null) {
        subs.debrief = `${(training.lastAccuracy * 100).toFixed(1)}% accuracy`;
      } else {
        subs.debrief = `step ${training.step.toLocaleString()}`;
      }
    } else if (architecture.layers.length > 0) {
      subs.debrief = 'ready to train';
    } else if (synthesis.loaded && symbolCount > 0) {
      subs.debrief = 'design needed';
    } else {
      subs.debrief = '';
    }

    return subs;
  });

  // Load synthesis state at app start so every tab can rely on it. Without
  // this, going Orientation → Architecture → Training (skipping the Data
  // Synthesis tab) leaves synthesis.loaded = false, which breaks Training
  // (no symbols selected, batch can't load).
  $effect(() => {
    if (synthesis.loaded) return;
    (async () => {
      try {
        const [{ categories }, { fonts }, intermediate] = await Promise.all([
          api.symbols(),
          api.fonts(),
          api.preset('intermediate'),
        ]);
        synthesis.categories = categories;
        synthesis.fonts = fonts;
        applyPreset(intermediate);
        synthesis.loaded = true;
      } catch (e) {
        console.error('synthesis preload failed', e);
      }
    })();
  });

  // Persist the active tab to localStorage on every change so a reload
  // returns the user to the same tab. The first run after page load just
  // re-saves whatever was restored at module init — cheap and keeps the
  // effect declarative.
  $effect(() => {
    // Reading ui.activeTab inside persistUiPrefs() registers it as a dep.
    persistUiPrefs();
  });

  // Auto-load on restart. Once synthesis is ready (so applyCheckpointResponse
  // can swap categories/fonts/augmentation in without racing the preload),
  // attempt to load the checkpoint named in the filename text box. Only
  // runs once per page load — guarded by a local flag.
  let autoLoadAttempted = false;
  $effect(() => {
    if (!synthesis.loaded) return;
    if (autoLoadAttempted) return;
    if (!training.autoLoadOnRestart) return;
    const fname = training.checkpointFilename.trim();
    if (!fname) return;
    autoLoadAttempted = true;
    (async () => {
      try {
        const { files } = await api.listCheckpoints();
        training.availableCheckpoints = files;
        const withExt = fname.endsWith('.pt') ? fname : `${fname}.pt`;
        if (!files.some((c) => c.name === withExt)) {
          console.info(
            `auto-load skipped: no checkpoint named ${withExt}`
          );
          return;
        }
        const result = await api.loadCheckpoint(fname);
        await applyCheckpointResponse(result);
      } catch (e) {
        console.warn('auto-load failed:', e);
      }
    })();
  });

  // Whenever the data synthesis config changes (categories, font usage,
  // augmentation), invalidate any architecture and training session
  // built against the old config — class identity / count is determined
  // by selectedCategories, so the existing model's class table won't
  // line up with new batches. JSON.stringify is enough to track the
  // deep state via Svelte's reactive proxies.
  //
  // The `suppressSynthesisInvalidation` flag lets the checkpoint-load
  // path rewrite synthesis state to match the loaded model without the
  // effect tearing the session back down. We still update the sig so
  // that once the suppression lifts, the new config is treated as the
  // baseline.
  let lastSynthesisSig: string | null = null;
  $effect(() => {
    const sig = JSON.stringify({
      cats: synthesis.selectedCategories,
      fonts: synthesis.fontUsage,
      aug: synthesis.augmentation,
    });
    if (!synthesis.loaded) return;
    if (lastSynthesisSig === null) {
      lastSynthesisSig = sig;
      return;
    }
    if (sig === lastSynthesisSig) return;
    lastSynthesisSig = sig;
    if (training.suppressSynthesisInvalidation > 0) return;

    // Architecture → empty layer list, no stale suggestion.
    architecture.layers = [];
    architecture.suggestionReasoning = null;

    // Training session state → fully reset. We also clear the batch
    // so that if the user is on a non-Training tab when the synthesis
    // config changes, returning to Training won't show a stale batch
    // (rendered against the old fonts/categories).
    training.hasSession = false;
    training.numClasses = 0;
    training.paramCount = 0;
    training.step = 0;
    training.lastLoss = null;
    training.lastAccuracy = null;
    training.lossHistory = [];
    training.valLossHistory = [];
    training.batch = [];
    training.predictions = [];
    training.batchVerdict = [];
    training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
    training.selectedIndex = null;
    training.animating = false;

    // Tell the backend so the next train_batch / eval doesn't hit a
    // stale session. Fire-and-forget — failures don't block the UI.
    api.resetTraining().catch((e) =>
      console.warn('training reset failed:', e)
    );
  });

  // ── Pipeline state sync (UI ↔ backend mirror ↔ agent) ───────────────
  //
  // Both the user (via Svelte stores) and the ML Engineer agent (via
  // MCP tools) need to read and write the same pipeline configuration.
  // The backend holds the source of truth; this block keeps the
  // frontend stores in lockstep with it.
  //
  // Two flows:
  //   1. WS /ws/state pushes patches whenever the agent (or any other
  //      tab) changes state. We apply them to the local stores.
  //   2. A reactive $effect computes a signature of the synthesis /
  //      architecture / training fields the backend mirror tracks. When
  //      the signature changes (i.e. the user moved a slider), we POST
  //      /api/state/patch with the relevant slice.
  //
  // Echo prevention: after applying a WS event we update
  // `lastSyncedSig` so the auto-sync effect sees no diff and skips the
  // POST. After our own POST we also update it pre-emptively so the
  // backend's broadcast (when it arrives) is recognized as our own.

  let lastSyncedSig: string | null = null;
  let stateLoaded = $state(false);
  let suppressLocalSync = false;

  function _layersWireFormat(): { type: string; params: Record<string, number> }[] {
    return architecture.layers.map((l) => ({ type: l.type, params: { ...l.params } }));
  }

  function _localStateForSync() {
    return {
      synthesis: {
        selectedCategories: { ...synthesis.selectedCategories },
        fontUsage: { ...synthesis.fontUsage },
        augmentation: {
          noise: { ...synthesis.augmentation.noise },
          skew: { ...synthesis.augmentation.skew },
        },
        activePreset: synthesis.activePreset,
      },
      architecture: {
        layers: _layersWireFormat(),
        hyperparameters: { ...architecture.hyperparameters },
      },
      training: {
        validateEveryN: training.validateEveryN,
        samplesPerSymbolPerEpoch: training.samplesPerSymbolPerEpoch,
      },
    };
  }

  // Apply a partial backend snapshot (or patch) to the local stores.
  // Each branch checks for presence so a partial patch leaves untouched
  // fields alone. Nested writes use `Object.assign` / spread so Svelte 5
  // notices the mutation and re-renders dependents.
  function _applyServerState(patch: Record<string, any>) {
    suppressLocalSync = true;
    try {
      if (patch.synthesis) {
        const s = patch.synthesis;
        if (s.selectedCategories) {
          synthesis.selectedCategories = { ...s.selectedCategories };
        }
        if (s.fontUsage) {
          synthesis.fontUsage = { ...s.fontUsage };
        }
        if (s.augmentation) {
          synthesis.augmentation = {
            noise: { ...synthesis.augmentation.noise, ...s.augmentation.noise },
            skew: { ...synthesis.augmentation.skew, ...s.augmentation.skew },
          };
        }
        if ('activePreset' in s) {
          synthesis.activePreset = s.activePreset;
        }
      }
      if (patch.architecture) {
        const a = patch.architecture;
        if (a.layers) {
          // Layer ids are frontend-only — generate fresh ones for
          // anything that came from the server.
          let counter = 0;
          const nid = () => `srv-${Date.now().toString(36)}-${counter++}`;
          architecture.layers = (a.layers as { type: string; params: Record<string, number> }[]).map(
            (l) => ({ id: nid(), type: l.type as LayerType, params: { ...l.params } }),
          );
        }
        if (a.hyperparameters) {
          architecture.hyperparameters = {
            lr: a.hyperparameters.lr ?? architecture.hyperparameters.lr,
            batch_size: a.hyperparameters.batch_size ?? architecture.hyperparameters.batch_size,
            optimizer:
              (a.hyperparameters.optimizer as Optimizer) ?? architecture.hyperparameters.optimizer,
          };
        }
      }
      if (patch.training) {
        const t = patch.training;
        if (typeof t.validateEveryN === 'number')
          training.validateEveryN = t.validateEveryN;
        if (typeof t.samplesPerSymbolPerEpoch === 'number')
          training.samplesPerSymbolPerEpoch = t.samplesPerSymbolPerEpoch;
      }
      lastSyncedSig = JSON.stringify(_localStateForSync());
    } finally {
      // Release on next microtask so the chained $effect runs once
      // and skips (it sees the updated lastSyncedSig).
      queueMicrotask(() => { suppressLocalSync = false; });
    }
  }

  // Initial state load: backend is authoritative on app boot. Wait for
  // synthesis.loaded so the frontend has the category/font catalogs to
  // render against, then merge the backend's last-seen state on top of
  // the synthesis preload defaults.
  let initialStateFetched = false;
  $effect(() => {
    if (!synthesis.loaded || initialStateFetched) return;
    initialStateFetched = true;
    (async () => {
      try {
        const snap = await api.getState();
        // Skip applying if the backend's mirror is "empty" (no categories
        // selected, no layers): leave the user's existing local state /
        // localStorage choices alone in that case so the first run is
        // pre-populated by applyPreset() rather than a blank canvas.
        const hasContent =
          Object.values(snap.synthesis.selectedCategories || {}).some(Boolean) ||
          (snap.architecture.layers && snap.architecture.layers.length > 0);
        if (hasContent) {
          _applyServerState(snap as unknown as Record<string, any>);
        } else {
          // Push current local state to seed the backend mirror.
          lastSyncedSig = JSON.stringify(_localStateForSync());
          api.patchState(_localStateForSync()).catch((e) =>
            console.warn('initial state push failed', e),
          );
        }
        stateLoaded = true;
      } catch (e) {
        console.warn('initial state fetch failed', e);
        stateLoaded = true; // proceed with local defaults
      }
    })();
  });

  // Auto-sync local edits to backend. Debounced so dragging a slider
  // doesn't fire 60 POSTs/sec.
  let syncTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    // Read all watched fields so they register as deps. Using the
    // helper means we only have to maintain the field list in one
    // place.
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
      api.patchState(_localStateForSync() as unknown as Record<string, unknown>).catch(
        (e) => console.warn('state patch push failed', e),
      );
    }, 200);
  });

  // Cap mirrors TrainingTab.svelte — keep agent-driven loss series the
  // same length as user-driven ones so the chart x-axis behaves the
  // same regardless of who's training.
  const MAX_LOSS_HISTORY = 2000;

  // WebSocket subscription. Reconnects automatically on disconnect.
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
          // Agent (re)built or dropped the live session. Mirror the
          // metadata + loss series into the training store so the
          // Training tab's step counter, param count, and charts
          // match the backend.
          if (ev.hasSession) {
            training.hasSession = true;
            if (typeof ev.numClasses === 'number')
              training.numClasses = ev.numClasses;
            if (typeof ev.paramCount === 'number')
              training.paramCount = ev.paramCount;
            if (typeof ev.step === 'number') training.step = ev.step;
            training.lossHistory = (ev.lossHistory ?? []).map((p) => ({
              ...p,
            }));
            training.valLossHistory = (ev.valLossHistory ?? []).map(
              (p) => ({ ...p }),
            );
            training.lastLoss =
              training.lossHistory.length > 0
                ? training.lossHistory[training.lossHistory.length - 1].loss
                : null;
            training.lastAccuracy = null;
            training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
            // Force a fresh batch on the user's next visit to the
            // Training tab — the existing one was rendered against
            // whatever synthesis was set before the agent rebuilt the
            // session, and predictions are now stale.
            training.batch = [];
            training.predictions = [];
            training.batchVerdict = [];
            training.selectedIndex = null;
          } else {
            training.hasSession = false;
            training.numClasses = 0;
            training.paramCount = 0;
            training.step = 0;
            training.lossHistory = [];
            training.valLossHistory = [];
            training.lastLoss = null;
            training.lastAccuracy = null;
            training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
          }
        } else if (ev.type === 'training_tick') {
          training.step = ev.step;
          training.lastLoss = ev.loss;
          training.lastAccuracy = ev.accuracy;
          training.lossHistory.push({ step: ev.step, loss: ev.loss });
          if (training.lossHistory.length > MAX_LOSS_HISTORY) {
            training.lossHistory =
              training.lossHistory.slice(-MAX_LOSS_HISTORY);
          }
        } else if (ev.type === 'validation_tick') {
          training.valLossHistory.push({ step: ev.step, loss: ev.loss });
          if (training.valLossHistory.length > MAX_LOSS_HISTORY) {
            training.valLossHistory =
              training.valLossHistory.slice(-MAX_LOSS_HISTORY);
          }
        }
      },
      () => {
        // Reconnect after a short backoff. Production-grade backoff
        // would scale up; this is enough for the dev/demo use case.
        setTimeout(connectStateWs, 1500);
      },
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
          <!-- Always render the subtitle row (even empty) so every tab
               has the same vertical extent — that keeps label baselines
               aligned across the row when the nav uses items-end. -->
          <span class="tab-subtitle">{tabSubtitles[tab.id] || ' '}</span>
        </span>
      </button>
    {/each}
  </nav>

  <div class="flex-1 min-h-0 flex overflow-hidden">
    <main class="flex-1 min-w-0 min-h-0 overflow-auto">
      {#if ui.activeTab === 'orientation'}
        <OrientationTab />
      {:else if ui.activeTab === 'data'}
        <DataSynthesisTab />
      {:else if ui.activeTab === 'architecture'}
        <ArchitectureTab />
      {:else if ui.activeTab === 'training'}
        <TrainingTab />
      {:else if ui.activeTab === 'inference'}
        <InferenceTab />
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

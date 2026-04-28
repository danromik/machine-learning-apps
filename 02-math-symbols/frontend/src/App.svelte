<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
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
    INPUT_SHAPE,
    applyPreset,
    applyCheckpointResponse,
    persistUiPrefs,
    type TabId,
  } from './state.svelte';
  import { api } from './api';
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

  <main class="flex-1 min-h-0 overflow-auto">
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

  <StatusBar />
</div>

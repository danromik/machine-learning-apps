<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import DataSynthesisTab from './components/tabs/DataSynthesisTab.svelte';
  import ArchitectureTab from './components/tabs/ArchitectureTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import InferenceTab from './components/tabs/InferenceTab.svelte';
  import {
    ui,
    synthesis,
    architecture,
    training,
    INPUT_SHAPE,
    applyPreset,
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

    return subs;
  });

  $effect(() => {
    (async () => {
      try {
        const { device } = await api.device();
        ui.device = device;
        ui.status = 'ready';
      } catch (e) {
        const msg = (e as Error).message || String(e);
        ui.status = `backend unreachable: ${msg} — is the server on :5041?`;
        console.error('initial fetch failed', e);
      }
    })();
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
</script>

<div class="h-screen w-screen flex flex-col overflow-hidden">
  <Header />

  <nav
    class="flex items-end gap-1 px-4 border-b border-[var(--color-border)]
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
    {/if}
  </main>

  <StatusBar />
</div>

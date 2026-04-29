<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import DataAcquisitionTab from './components/tabs/DataAcquisitionTab.svelte';
  import ArchitectureTab from './components/tabs/ArchitectureTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import InferenceTab from './components/tabs/InferenceTab.svelte';
  import DebriefTab from './components/tabs/DebriefTab.svelte';
  import {
    ui,
    dataset,
    architecture,
    training,
    INPUT_SHAPE,
    applyCheckpointResponse,
    persistUiPrefs,
    isDatasetReady,
    type TabId,
  } from './state.svelte';
  import { api } from './api';
  import {
    computeArchitecture,
    formatCount,
  } from './components/tabs/architecture/computeArchitecture';

  const TABS: { id: TabId; label: string }[] = [
    { id: 'orientation', label: 'Orientation' },
    { id: 'data', label: 'Data Acquisition' },
    { id: 'architecture', label: 'Model Architecture' },
    { id: 'training', label: 'Training' },
    { id: 'inference', label: 'Inference' },
    { id: 'debrief', label: 'Debrief' },
  ];

  let datasetReady = $derived(isDatasetReady());

  let totalParams = $derived.by(() => {
    if (architecture.preset === 'resnet18') return training.paramCount || 0;
    const numClasses = dataset.classes.length || 10;
    return computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
      .totalParams;
  });

  // One epoch = enough batches to see each training image once on
  // average. With 10 classes × ~950 train/class ≈ 9500 imgs and a batch
  // of 64, that's ~150 batches per epoch.
  let batchesPerEpoch = $derived.by(() => {
    const totalTrain = dataset.status?.num_train ?? 0;
    const bs = architecture.hyperparameters.batch_size;
    if (totalTrain <= 0 || bs <= 0) return 0;
    return Math.max(1, Math.ceil(totalTrain / bs));
  });

  let layerCountForSubtitle = $derived(
    architecture.preset === 'resnet18' ? 18 : architecture.layers.length
  );

  let tabSubtitles = $derived.by(() => {
    const subs: Record<TabId, string> = {
      orientation: '',
      data: dataset.loaded
        ? datasetReady
          ? `${(dataset.status?.num_train ?? 0).toLocaleString()} train · ` +
            `${(dataset.status?.num_val ?? 0).toLocaleString()} val`
          : 'not downloaded'
        : '',
      architecture: architecture.preset
        ? `${architecture.preset} · ${formatCount(totalParams)} weights`
        : `${layerCountForSubtitle} layers · ${formatCount(totalParams)} weights`,
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

    if (training.hasSession && training.step > 0) {
      if (training.lastAccuracy !== null) {
        subs.debrief = `${(training.lastAccuracy * 100).toFixed(1)}% accuracy`;
      } else {
        subs.debrief = `step ${training.step.toLocaleString()}`;
      }
    } else if (architecture.layers.length > 0 || architecture.preset) {
      subs.debrief = 'ready to train';
    } else if (datasetReady) {
      subs.debrief = 'design needed';
    } else {
      subs.debrief = '';
    }

    return subs;
  });

  // Load dataset metadata + class table + presets at app start so every
  // tab can rely on them being present.
  $effect(() => {
    if (dataset.loaded) return;
    (async () => {
      try {
        const [{ classes, input_size }, status, { presets }] = await Promise.all([
          api.classes(),
          api.datasetStatus(),
          api.architecturePresets(),
        ]);
        dataset.classes = classes;
        dataset.inputSize = input_size;
        dataset.status = status;
        architecture.presets = presets;
        dataset.loaded = true;
      } catch (e) {
        console.error('dataset preload failed', e);
      }
    })();
  });

  $effect(() => {
    persistUiPrefs();
  });

  // Auto-load on restart. Once dataset metadata is ready, attempt to load
  // the checkpoint named in the filename text box. Only runs once per
  // page load.
  let autoLoadAttempted = false;
  $effect(() => {
    if (!dataset.loaded) return;
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
          console.info(`auto-load skipped: no checkpoint named ${withExt}`);
          return;
        }
        const result = await api.loadCheckpoint(fname);
        await applyCheckpointResponse(result);
      } catch (e) {
        console.warn('auto-load failed:', e);
      }
    })();
  });

  // Whenever the dataset augmentation config changes, invalidate any
  // training session built against it (the input distribution changes,
  // so a model mid-training would be evaluating on shifted samples).
  // Suppress flag lets the checkpoint-load path rewrite augmentation
  // without tearing down the session it's restoring.
  let lastDatasetSig: string | null = null;
  $effect(() => {
    const sig = JSON.stringify({ aug: dataset.augmentation });
    if (!dataset.loaded) return;
    if (lastDatasetSig === null) {
      lastDatasetSig = sig;
      return;
    }
    if (sig === lastDatasetSig) return;
    lastDatasetSig = sig;
    if (training.suppressDatasetInvalidation > 0) return;

    // Augmentation changes don't change the *class set* (always 10) or
    // input shape, so we only need to refresh the current batch — not
    // tear down the architecture or session. Just clear the in-memory
    // batch so the Training tab fetches a fresh one with the new aug.
    training.batch = [];
    training.predictions = [];
    training.batchVerdict = [];
    training.selectedIndex = null;
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

  <main class="flex-1 min-h-0 overflow-auto">
    {#if ui.activeTab === 'orientation'}
      <OrientationTab />
    {:else if ui.activeTab === 'data'}
      <DataAcquisitionTab />
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

<script lang="ts">
  import { onMount } from 'svelte';
  import {
    architecture,
    synthesis,
    training,
  } from '../../state.svelte';
  import { api, streamSamples, type SynthesisSample } from '../../api';
  import HyperparametersBar from './training/HyperparametersBar.svelte';
  import Sidebar from './training/Sidebar.svelte';
  import BatchGrid from './training/BatchGrid.svelte';
  import PredictionBar from './training/PredictionBar.svelte';

  let busy = $state(false);
  let statusMsg = $state<string>('idle');
  // Plain `let`, NOT $state — buildSampleRequest does `++batchSeed`, and if
  // batchSeed were reactive the read + write inside the $effect's tracked
  // call chain would invalidate the effect on every run, causing an infinite
  // loop that crashes the Svelte runtime (and breaks tab switching).
  let batchSeed = 0;
  let abortController: AbortController | null = null;

  // Flat list of currently-selected class symbols, in (category-order ×
  // symbol-order). The backend uses this exact ordering for label indexing.
  let selectedClasses = $derived.by(() => {
    if (!synthesis.loaded) return [] as string[];
    const out: string[] = [];
    for (const c of synthesis.categories) {
      if (synthesis.selectedCategories[c.id]) out.push(...c.symbols);
    }
    return out;
  });

  // Reinit prerequisites — surfaced both as a disabled state and as a
  // visible reason in the Sidebar so the user isn't left guessing.
  let reinitBlockedReason = $derived.by(() => {
    if (!synthesis.loaded) return 'Loading data synthesis…';
    if (selectedClasses.length === 0)
      return 'Select at least one symbol category in Data Synthesis.';
    if (architecture.layers.length === 0)
      return 'Build (or Suggest) an architecture in Model Architecture.';
    return '';
  });
  let canReinit = $derived(reinitBlockedReason === '');

  // ── lifecycle ───────────────────────────────────────────────────────────

  onMount(async () => {
    // Fetch initial state + checkpoint list
    try {
      const [state, ckpts] = await Promise.all([
        api.trainingState(),
        api.listCheckpoints(),
      ]);
      training.availableCheckpoints = ckpts.files;
      if (state.has_session) {
        training.hasSession = true;
        training.numClasses = state.num_classes;
        training.paramCount = state.param_count;
        training.step = state.step;
      } else {
        training.hasSession = false;
      }
      // Load the initial batch immediately so the user always sees something.
      await loadBatch();
    } catch (e) {
      statusMsg = `init failed: ${(e as Error).message}`;
    }
  });

  // Reload when batch_size changes or synthesis becomes available.
  // We deliberately don't read `busy` here: the AbortController inside
  // loadBatch serializes overlapping loads, and trainOneBatch loads its
  // own follow-up batch — so a busy-driven re-run is just redundant work.
  $effect(() => {
    void architecture.hyperparameters.batch_size;
    void synthesis.loaded;
    loadBatch();
  });

  // ── batch + predict ────────────────────────────────────────────────────

  async function loadBatch() {
    if (!synthesis.loaded) return;
    abortController?.abort();
    abortController = new AbortController();
    const cfg = buildSampleRequest(architecture.hyperparameters.batch_size);
    if (cfg.training_fonts.length === 0 || cfg.categories.length === 0) {
      training.batch = [];
      training.predictions = [];
      training.selectedIndex = null;
      return;
    }
    const samples: SynthesisSample[] = [];
    try {
      for await (const s of streamSamples(cfg, abortController.signal)) {
        samples.push(s);
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      statusMsg = `batch load failed: ${(e as Error).message}`;
      return;
    }
    training.batch = samples;
    training.selectedIndex = null;
    training.predictions = [];
    if (training.hasSession) await refreshPredictions();
  }

  function buildSampleRequest(count: number) {
    return {
      categories: synthesis.categories
        .filter((c) => synthesis.selectedCategories[c.id])
        .map((c) => c.id),
      training_fonts: synthesis.fonts
        .filter((f) => synthesis.fontUsage[f.family] === 'train')
        .map((f) => f.family),
      validation_fonts: synthesis.fonts
        .filter((f) => synthesis.fontUsage[f.family] === 'val')
        .map((f) => f.family),
      augmentation: {
        noise: { ...synthesis.augmentation.noise },
        skew: { ...synthesis.augmentation.skew },
      },
      split: 'train' as const,
      count,
      seed: ++batchSeed * 1000 + Date.now() % 1000,
    };
  }

  async function refreshPredictions() {
    if (training.batch.length === 0 || !training.hasSession) return;
    try {
      const { predictions } = await api.predict(
        training.batch.map((b) => b.png_b64)
      );
      training.predictions = predictions;
    } catch (e) {
      statusMsg = `predict failed: ${(e as Error).message}`;
    }
  }

  // ── actions ────────────────────────────────────────────────────────────

  async function reinitialize() {
    if (selectedClasses.length === 0) {
      statusMsg = 'no symbols selected — configure Data Synthesis first';
      return;
    }
    if (architecture.layers.length === 0) {
      statusMsg = 'no architecture defined — design or suggest one first';
      return;
    }
    busy = true;
    statusMsg = 'building model…';
    try {
      const result = await api.initTraining({
        architecture: architecture.layers.map((l) => ({
          type: l.type,
          params: { ...l.params },
        })),
        hyperparameters: {
          lr: architecture.hyperparameters.lr,
          batch_size: architecture.hyperparameters.batch_size,
          optimizer: architecture.hyperparameters.optimizer,
        },
        classes: selectedClasses,
      });
      training.hasSession = true;
      training.numClasses = result.num_classes;
      training.paramCount = result.param_count;
      training.step = result.step;
      statusMsg = `model initialized · ${result.param_count.toLocaleString()} params · ${result.num_classes} classes`;
      await refreshPredictions();
    } catch (e) {
      statusMsg = `init failed: ${(e as Error).message}`;
    } finally {
      busy = false;
    }
  }

  async function trainOneBatch() {
    if (training.batch.length === 0 || !training.hasSession) return;
    busy = true;
    training.animating = true;
    statusMsg = 'training (1 batch)…';
    try {
      // Make sure we have predictions to animate through.
      if (training.predictions.length !== training.batch.length) {
        await refreshPredictions();
      }
      // Step through each image with a delay long enough to perceive the
      // highlight + bar chart for that sample. 50–100 ms per image keeps
      // the sweep visible without dragging on for too long: a 128-batch
      // takes ~6.4s, a 32-batch ~3.2s.
      const total = training.batch.length;
      const stepDelay = Math.max(50, Math.min(100, Math.floor(4000 / total)));
      for (let i = 0; i < total; i++) {
        training.selectedIndex = i;
        await sleep(stepDelay);
      }
      // Apply the actual gradient step on the same batch.
      const result = await api.trainBatch(
        training.batch.map((b) => b.png_b64),
        training.batch.map((b) => b.label)
      );
      training.lastLoss = result.loss;
      training.step = result.step;
      statusMsg = `step ${result.step} · loss ${result.loss.toFixed(4)}`;
      // Load a new batch + new predictions for the next round.
      await loadBatch();
    } catch (e) {
      statusMsg = `train failed: ${(e as Error).message}`;
    } finally {
      training.animating = false;
      busy = false;
    }
  }

  async function saveCheckpoint() {
    busy = true;
    try {
      const { name } = await api.saveCheckpoint(training.checkpointFilename);
      training.availableCheckpoints = (
        await api.listCheckpoints()
      ).files;
      statusMsg = `saved ${name}`;
    } catch (e) {
      statusMsg = `save failed: ${(e as Error).message}`;
    } finally {
      busy = false;
    }
  }

  async function loadCheckpoint() {
    busy = true;
    try {
      const result = await api.loadCheckpoint(training.checkpointFilename);
      training.hasSession = true;
      training.numClasses = result.num_classes;
      training.paramCount = result.param_count;
      training.step = result.step;
      statusMsg = `loaded ${training.checkpointFilename} · step ${result.step}`;
      await refreshPredictions();
    } catch (e) {
      statusMsg = `load failed: ${(e as Error).message}`;
    } finally {
      busy = false;
    }
  }

  function onSelectImage(i: number) {
    training.selectedIndex = i;
  }

  function sleep(ms: number) {
    return new Promise((r) => setTimeout(r, ms));
  }
</script>

<div class="h-full flex flex-col">
  <!-- Top: Hyperparameters section -->
  <section class="px-4 pt-3 pb-2 border-b border-[var(--color-border)]">
    <h3 class="text-sm font-semibold text-[var(--color-heading)] mb-2">
      Hyperparameters
    </h3>
    <HyperparametersBar />
  </section>

  <!-- Middle: 3-column layout (sidebar | batch grid | prediction bar) -->
  <div class="flex-1 min-h-0 flex">
    <Sidebar
      onReinitialize={reinitialize}
      onTrainBatch={trainOneBatch}
      onSaveCheckpoint={saveCheckpoint}
      onLoadCheckpoint={loadCheckpoint}
      canTrain={training.hasSession}
      {canReinit}
      {reinitBlockedReason}
      {busy}
    />

    <div class="w-px shrink-0 bg-[var(--color-border)]"></div>

    <div class="flex-1 min-w-0 min-h-0 flex flex-col">
      <header
        class="flex items-baseline justify-between gap-3 px-3 py-1.5
               border-b border-[var(--color-border)]"
      >
        <h4 class="text-xs font-semibold text-[var(--color-heading)]">
          Batch · {training.batch.length}
        </h4>
        <span class="text-xs text-[var(--color-muted)] font-mono truncate">
          {#if training.hasSession}
            step {training.step.toLocaleString()} ·
            {training.paramCount.toLocaleString()} params ·
            {training.numClasses} classes
            {#if training.lastLoss !== null}
              · loss {training.lastLoss.toFixed(4)}
            {/if}
          {:else}
            no session
          {/if}
        </span>
      </header>
      <BatchGrid onSelect={onSelectImage} />
      <footer
        class="px-3 py-1.5 border-t border-[var(--color-border)]
               text-xs text-[var(--color-muted)] truncate"
      >
        {statusMsg}
      </footer>
    </div>

    <div class="w-px shrink-0 bg-[var(--color-border)]"></div>

    <div class="w-72 shrink-0 min-h-0 flex flex-col">
      <PredictionBar classes={selectedClasses} />
    </div>
  </div>
</div>

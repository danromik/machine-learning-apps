<script lang="ts">
  import { onMount } from 'svelte';
  import {
    architecture,
    dataset,
    training,
    applyCheckpointResponse,
    persistCheckpointPrefs,
    classLabels,
    isDatasetReady,
  } from '../../state.svelte';
  import { api, streamSamples, type Sample, type SampleRequest } from '../../api';
  import HyperparametersBar from './training/HyperparametersBar.svelte';
  import Sidebar from './training/Sidebar.svelte';
  import BatchGrid from './training/BatchGrid.svelte';
  import PredictionBar from './training/PredictionBar.svelte';
  import BatchChart from './training/BatchChart.svelte';
  import LossChart from './training/LossChart.svelte';

  let batchesPerEpoch = $derived.by(() => {
    const totalTrain = dataset.status?.num_train ?? 0;
    const bs = architecture.hyperparameters.batch_size;
    if (totalTrain <= 0 || bs <= 0) return 0;
    return Math.max(1, Math.ceil(totalTrain / bs));
  });

  // Plain `let`, not $state — buildSampleRequest mutates this on every
  // call; making it reactive would invalidate effects that read it.
  let batchSeed = 0;
  let abortController: AbortController | null = null;

  // Class table the model is trained against — fixed: 10 Imagenette
  // labels in index order, sourced from the dataset state.
  let classes = $derived.by(() => classLabels());

  let datasetReady = $derived(isDatasetReady());

  // Reinit prerequisites — surfaced as the disabled state and a visible
  // reason in the Sidebar. ResNet-18's locked preset has no layers so it
  // would otherwise look like "no architecture defined" — special-case
  // that to allow init.
  let reinitBlockedReason = $derived.by(() => {
    if (!dataset.loaded) return 'Loading dataset metadata…';
    if (!datasetReady)
      return 'Download the dataset in the Data Acquisition tab first.';
    if (architecture.preset !== 'resnet18' && architecture.layers.length === 0)
      return 'Pick a preset (or build one in Model Architecture).';
    return '';
  });
  let canReinit = $derived(reinitBlockedReason === '');

  // ── lifecycle ─────────────────────────────────────────────────────────

  onMount(async () => {
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
        training.preset = state.preset;
      } else {
        training.hasSession = false;
      }
    } catch (e) {
      training.statusMsg = `init failed: ${(e as Error).message}`;
    }
  });

  // (Re)load batch when batch_size or augmentation changes — but skip
  // on the first run after mount if the global batch is already populated
  // (preserved across tab unmount/remount).
  let lastBatchConfigSig: string | null = null;
  $effect(() => {
    if (!datasetReady) return;
    const sig = JSON.stringify({
      batch_size: architecture.hyperparameters.batch_size,
      aug: dataset.augmentation,
    });
    const isFirstRun = lastBatchConfigSig === null;
    const configChanged = !isFirstRun && sig !== lastBatchConfigSig;
    lastBatchConfigSig = sig;
    if (isFirstRun && training.batch.length > 0) {
      if (training.hasSession && training.predictions.length === 0) {
        void refreshPredictions();
      }
      return;
    }
    if (!isFirstRun && !configChanged) return;
    loadBatch();
  });

  // ── batch + predict ──────────────────────────────────────────────────

  function buildSampleRequest(count: number, split: 'train' | 'val' = 'train'): SampleRequest {
    return {
      split,
      count,
      seed: ++batchSeed * 1000 + (Date.now() % 1000),
      flip: dataset.augmentation.flip,
      jitter: dataset.augmentation.jitter,
      random_crop: dataset.augmentation.random_crop,
    };
  }

  async function loadBatch() {
    if (!datasetReady) return;
    abortController?.abort();
    abortController = new AbortController();
    const req = buildSampleRequest(architecture.hyperparameters.batch_size);
    const samples: Sample[] = [];
    try {
      for await (const s of streamSamples(req, abortController.signal)) {
        samples.push(s);
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      training.statusMsg = `batch load failed: ${(e as Error).message}`;
      return;
    }
    training.batch = samples;
    training.selectedIndex = null;
    training.predictions = [];
    training.batchVerdict = new Array(samples.length).fill(null);
    if (training.hasSession) await refreshPredictions();
  }

  async function refreshPredictions() {
    if (training.batch.length === 0 || !training.hasSession) return;
    try {
      const { predictions } = await api.predict(
        training.batch.map((b) => b.png_b64)
      );
      training.predictions = predictions;
    } catch (e) {
      training.statusMsg = `predict failed: ${(e as Error).message}`;
    }
  }

  // ── loss curves ──────────────────────────────────────────────────────

  const MAX_LOSS_HISTORY = 2000;

  function recordTrainLoss(step: number, loss: number) {
    training.lossHistory.push({ step, loss });
    if (training.lossHistory.length > MAX_LOSS_HISTORY) {
      training.lossHistory = training.lossHistory.slice(-MAX_LOSS_HISTORY);
    }
  }

  async function maybeRunValidation(step: number) {
    if (!training.hasSession) return;
    if (training.validateEveryN <= 0 || step <= 0) return;
    if (step % training.validateEveryN !== 0) return;
    const req = buildSampleRequest(architecture.hyperparameters.batch_size, 'val');
    try {
      const samples: Sample[] = [];
      for await (const s of streamSamples(req)) samples.push(s);
      if (samples.length === 0) return;
      const r = await api.evalBatch(
        samples.map((s) => s.png_b64),
        samples.map((s) => s.label)
      );
      training.valLossHistory.push({ step, loss: r.loss });
      if (training.valLossHistory.length > MAX_LOSS_HISTORY) {
        training.valLossHistory = training.valLossHistory.slice(-MAX_LOSS_HISTORY);
      }
    } catch (e) {
      console.warn('val skipped:', e);
    }
  }

  // ── actions ──────────────────────────────────────────────────────────

  async function reinitialize() {
    if (!datasetReady) {
      training.statusMsg = 'dataset not downloaded';
      return;
    }
    if (architecture.preset !== 'resnet18' && architecture.layers.length === 0) {
      training.statusMsg = 'no architecture defined — pick a preset first';
      return;
    }
    training.busy = true;
    training.statusMsg = 'building model…';
    try {
      const result = await api.initTraining({
        architecture: architecture.layers.map((l) => ({
          type: l.type,
          params: { ...l.params },
        })),
        preset: architecture.preset,
        hyperparameters: {
          lr: architecture.hyperparameters.lr,
          batch_size: architecture.hyperparameters.batch_size,
          optimizer: architecture.hyperparameters.optimizer,
        },
        classes,
        dataset_config: {
          augmentation: { ...dataset.augmentation },
        },
      });
      training.hasSession = true;
      training.numClasses = result.num_classes;
      training.paramCount = result.param_count;
      training.step = result.step;
      training.preset = result.preset;
      training.lastLoss = null;
      training.lastAccuracy = null;
      training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
      training.lossHistory = [];
      training.valLossHistory = [];
      training.statusMsg = `model initialized · ${result.param_count.toLocaleString()} params · ${result.num_classes} classes`;
      await refreshPredictions();
    } catch (e) {
      training.statusMsg = `init failed: ${(e as Error).message}`;
    } finally {
      training.busy = false;
    }
  }

  async function trainOneBatch() {
    if (training.batch.length === 0 || !training.hasSession) return;
    training.busy = true;
    training.animating = true;
    training.statusMsg = 'training (1 batch)…';
    try {
      if (training.predictions.length !== training.batch.length) {
        await refreshPredictions();
      }
      const total = training.batch.length;
      const stepDelay = Math.max(50, Math.min(100, Math.floor(4000 / total)));
      training.batchVerdict = new Array(total).fill(null);
      training.batchChartCounts = { correct: 0, incorrect: 0, total };
      for (let i = 0; i < total; i++) {
        training.selectedIndex = i;
        await sleep(stepDelay);
        recordVerdict(i);
        if (training.batchVerdict[i] === 'correct')
          training.batchChartCounts.correct++;
        else if (training.batchVerdict[i] === 'incorrect')
          training.batchChartCounts.incorrect++;
      }
      const result = await api.trainBatch(
        training.batch.map((b) => b.png_b64),
        training.batch.map((b) => b.label),
        architecture.hyperparameters.lr,
        architecture.hyperparameters.optimizer
      );
      training.lastLoss = result.loss;
      training.lastAccuracy = result.accuracy;
      training.step = result.step;
      recordTrainLoss(result.step, result.loss);
      await maybeRunValidation(result.step);
      training.statusMsg =
        `step ${result.step} · loss ${result.loss.toFixed(4)} ` +
        `· acc ${(result.accuracy * 100).toFixed(1)}%`;
      await loadBatch();
    } catch (e) {
      training.statusMsg = `train failed: ${(e as Error).message}`;
    } finally {
      training.animating = false;
      training.busy = false;
      await maybeAutoSave();
    }
  }

  async function trainOneEpoch() {
    if (!training.hasSession || batchesPerEpoch <= 0) return;
    if (training.batch.length === 0) {
      training.statusMsg = 'no batch loaded';
      return;
    }
    training.busy = true;
    training.epochRunning = true;
    training.abortEpoch = false;
    const total = batchesPerEpoch;
    training.statusMsg = `training (1 epoch = ${total} batches)…`;
    try {
      for (let i = 0; i < total; i++) {
        if (training.abortEpoch) {
          training.statusMsg = `epoch interrupted (batch ${i}/${total}) · step ${training.step}`;
          return;
        }
        if (training.batch.length === 0) {
          training.statusMsg = `epoch stopped: failed to load batch ${i + 1}/${total}`;
          return;
        }
        const result = await api.trainBatch(
          training.batch.map((b) => b.png_b64),
          training.batch.map((b) => b.label),
          architecture.hyperparameters.lr,
          architecture.hyperparameters.optimizer
        );
        training.lastLoss = result.loss;
        training.lastAccuracy = result.accuracy;
        training.step = result.step;
        recordTrainLoss(result.step, result.loss);
        await maybeRunValidation(result.step);
        training.statusMsg =
          `epoch (batch ${i + 1}/${total}) · step ${result.step} ` +
          `· loss ${result.loss.toFixed(4)} ` +
          `· acc ${(result.accuracy * 100).toFixed(1)}%`;
        snapshotBatchChartCounts();
        await loadBatch();
      }
      training.statusMsg =
        `epoch complete · step ${training.step} ` +
        (training.lastLoss !== null
          ? `· loss ${training.lastLoss.toFixed(4)} `
          : '') +
        (training.lastAccuracy !== null
          ? `· acc ${(training.lastAccuracy * 100).toFixed(1)}%`
          : '');
    } catch (e) {
      training.statusMsg = `epoch train failed: ${(e as Error).message}`;
    } finally {
      training.abortEpoch = false;
      training.epochRunning = false;
      training.busy = false;
      await maybeAutoSave();
    }
  }

  function stopEpoch() {
    training.abortEpoch = true;
  }

  async function trainOneBatchFast() {
    if (training.batch.length === 0 || !training.hasSession) return;
    training.busy = true;
    training.statusMsg = 'training (1 batch)…';
    try {
      const result = await api.trainBatch(
        training.batch.map((b) => b.png_b64),
        training.batch.map((b) => b.label),
        architecture.hyperparameters.lr,
        architecture.hyperparameters.optimizer
      );
      training.lastLoss = result.loss;
      training.lastAccuracy = result.accuracy;
      training.step = result.step;
      recordTrainLoss(result.step, result.loss);
      await maybeRunValidation(result.step);
      training.statusMsg =
        `step ${result.step} · loss ${result.loss.toFixed(4)} ` +
        `· acc ${(result.accuracy * 100).toFixed(1)}%`;
      snapshotBatchChartCounts();
      await loadBatch();
    } catch (e) {
      training.statusMsg = `train failed: ${(e as Error).message}`;
    } finally {
      training.busy = false;
      await maybeAutoSave();
    }
  }

  async function trainContinuously() {
    if (!training.hasSession) return;
    if (training.batch.length === 0) {
      training.statusMsg = 'no batch loaded';
      return;
    }
    training.busy = true;
    training.continuousRunning = true;
    training.abortContinuous = false;
    training.statusMsg = 'training continuously…';
    try {
      while (!training.abortContinuous) {
        if (training.batch.length === 0) {
          training.statusMsg = 'continuous stopped: failed to load batch';
          return;
        }
        const result = await api.trainBatch(
          training.batch.map((b) => b.png_b64),
          training.batch.map((b) => b.label),
          architecture.hyperparameters.lr,
          architecture.hyperparameters.optimizer
        );
        training.lastLoss = result.loss;
        training.lastAccuracy = result.accuracy;
        training.step = result.step;
        recordTrainLoss(result.step, result.loss);
        await maybeRunValidation(result.step);
        const epochsDone =
          batchesPerEpoch > 0
            ? Math.floor(result.step / batchesPerEpoch)
            : 0;
        training.statusMsg =
          `continuous · step ${result.step} ` +
          `(${epochsDone} epoch${epochsDone === 1 ? '' : 's'}) ` +
          `· loss ${result.loss.toFixed(4)} ` +
          `· acc ${(result.accuracy * 100).toFixed(1)}%`;
        snapshotBatchChartCounts();
        await loadBatch();
      }
      training.statusMsg =
        `stopped · step ${training.step} ` +
        (training.lastLoss !== null
          ? `· loss ${training.lastLoss.toFixed(4)} `
          : '') +
        (training.lastAccuracy !== null
          ? `· acc ${(training.lastAccuracy * 100).toFixed(1)}%`
          : '');
    } catch (e) {
      training.statusMsg = `continuous train failed: ${(e as Error).message}`;
    } finally {
      training.abortContinuous = false;
      training.continuousRunning = false;
      training.busy = false;
      await maybeAutoSave();
    }
  }

  function stopContinuous() {
    training.abortContinuous = true;
  }

  async function saveCheckpoint() {
    training.busy = true;
    try {
      const { name } = await api.saveCheckpoint(training.checkpointFilename);
      training.availableCheckpoints = (await api.listCheckpoints()).files;
      training.statusMsg = `saved ${name}`;
    } catch (e) {
      training.statusMsg = `save failed: ${(e as Error).message}`;
    } finally {
      training.busy = false;
    }
  }

  async function maybeAutoSave() {
    if (!training.autoSave) return;
    if (!training.hasSession) return;
    const fname = training.checkpointFilename.trim();
    if (!fname) return;
    try {
      const { name } = await api.saveCheckpoint(fname);
      training.availableCheckpoints = (await api.listCheckpoints()).files;
      training.statusMsg = `${training.statusMsg} · auto-saved ${name}`;
    } catch (e) {
      console.warn('auto-save failed:', e);
    }
  }

  function onToggleAutoSave(v: boolean) {
    training.autoSave = v;
    persistCheckpointPrefs();
  }

  function onToggleAutoLoadOnRestart(v: boolean) {
    training.autoLoadOnRestart = v;
    persistCheckpointPrefs();
  }

  function onChangeFilename(v: string) {
    training.checkpointFilename = v;
    persistCheckpointPrefs();
  }

  async function deleteCheckpointFile(name: string) {
    training.busy = true;
    try {
      await api.deleteCheckpoint(name);
      training.availableCheckpoints = (await api.listCheckpoints()).files;
      training.statusMsg = `deleted ${name}`;
    } catch (e) {
      training.statusMsg = `delete failed: ${(e as Error).message}`;
    } finally {
      training.busy = false;
    }
  }

  async function loadCheckpoint() {
    training.busy = true;
    try {
      const result = await api.loadCheckpoint(training.checkpointFilename);
      await applyCheckpointResponse(result);
      await loadBatch();
      training.statusMsg = `loaded ${training.checkpointFilename} · step ${result.step}`;
    } catch (e) {
      training.statusMsg = `load failed: ${(e as Error).message}`;
    } finally {
      training.busy = false;
    }
  }

  function onSelectImage(i: number) {
    training.selectedIndex = i;
    recordVerdict(i);
  }

  function recordVerdict(i: number) {
    const probs = training.predictions[i];
    const sample = training.batch[i];
    if (!probs || !sample) return;
    let bestIdx = 0;
    for (let k = 1; k < probs.length; k++) {
      if (probs[k] > probs[bestIdx]) bestIdx = k;
    }
    const predicted = classes[bestIdx];
    training.batchVerdict[i] =
      predicted === sample.label ? 'correct' : 'incorrect';
  }

  function snapshotBatchChartCounts() {
    let correct = 0;
    let incorrect = 0;
    for (let i = 0; i < training.batch.length; i++) {
      const probs = training.predictions[i];
      const sample = training.batch[i];
      if (!probs || !sample) continue;
      let bestIdx = 0;
      for (let k = 1; k < probs.length; k++) {
        if (probs[k] > probs[bestIdx]) bestIdx = k;
      }
      if (classes[bestIdx] === sample.label) correct++;
      else incorrect++;
    }
    training.batchChartCounts = {
      correct,
      incorrect,
      total: training.batch.length,
    };
  }

  function sleep(ms: number) {
    return new Promise((r) => setTimeout(r, ms));
  }
</script>

<div class="h-full flex flex-col">
  <section class="px-4 pt-3 pb-2 border-b border-[var(--color-border)]">
    <h3 class="text-sm font-semibold text-[var(--color-heading)] mb-2">
      Hyperparameters
    </h3>
    <HyperparametersBar />
  </section>

  <div class="flex-1 min-h-0 flex">
    <Sidebar
      onReinitialize={reinitialize}
      onTrainBatchFun={trainOneBatch}
      onTrainBatchFast={trainOneBatchFast}
      onTrainEpoch={trainOneEpoch}
      onStopEpoch={stopEpoch}
      onTrainContinuously={trainContinuously}
      onStopContinuous={stopContinuous}
      onSaveCheckpoint={saveCheckpoint}
      onLoadCheckpoint={loadCheckpoint}
      onChangeFilename={onChangeFilename}
      onDeleteCheckpoint={deleteCheckpointFile}
      onToggleAutoSave={onToggleAutoSave}
      onToggleAutoLoadOnRestart={onToggleAutoLoadOnRestart}
      canTrain={training.hasSession}
      {canReinit}
      {reinitBlockedReason}
      busy={training.busy}
      {batchesPerEpoch}
      epochRunning={training.epochRunning}
      continuousRunning={training.continuousRunning}
    />

    <div class="w-px shrink-0 bg-[var(--color-border)]"></div>

    <div class="flex-1 min-w-0 min-h-0 flex flex-col">
      <header
        class="flex items-baseline justify-between gap-3 px-3 py-1.5
               border-b border-[var(--color-border)]"
      >
        <h4 class="text-xs font-semibold text-[var(--color-heading)]">
          Next Batch · {training.batch.length}
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
        {training.statusMsg}
      </footer>
    </div>

    <div class="w-px shrink-0 bg-[var(--color-border)]"></div>

    <div class="w-96 shrink-0 min-h-0 flex flex-col">
      <header
        class="flex items-baseline justify-between gap-3 px-3 py-1.5
               border-b border-[var(--color-border)]"
      >
        <h4 class="text-xs font-semibold text-[var(--color-heading)]">
          Prediction
        </h4>
        <span class="text-[10px] text-[var(--color-muted)] font-mono">
          per-sample · per-batch
        </span>
      </header>
      <div class="flex-1 min-h-0 flex">
        <PredictionBar {classes} />
        <div class="w-px shrink-0 bg-[var(--color-border)]"></div>
        <div class="w-24 shrink-0 min-h-0 flex flex-col">
          <BatchChart />
        </div>
      </div>

      <header
        class="flex items-baseline justify-between gap-3 px-3 py-1.5
               border-y border-[var(--color-border)]"
      >
        <h4 class="text-xs font-semibold text-[var(--color-heading)]">
          Loss
        </h4>
        <span class="text-[10px] text-[var(--color-muted)] font-mono">
          validation every {training.validateEveryN} batches
        </span>
      </header>
      <div class="flex-1 min-h-0 flex">
        <LossChart
          data={training.lossHistory}
          color="var(--color-accent)"
          label="Training"
        />
        <div class="w-px shrink-0 bg-[var(--color-border)]"></div>
        <LossChart
          data={training.valLossHistory}
          color="var(--color-success)"
          label="Validation"
        />
      </div>
    </div>
  </div>
</div>

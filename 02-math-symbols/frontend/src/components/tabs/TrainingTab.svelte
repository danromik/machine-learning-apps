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
  import BatchChart from './training/BatchChart.svelte';
  import LossChart from './training/LossChart.svelte';

  let busy = $state(false);
  let statusMsg = $state<string>('idle');
  // True while trainOneEpoch is actively iterating. The sidebar uses this
  // to flip the Train (1 Epoch) button into a Stop button.
  let epochRunning = $state(false);
  // Set true to interrupt an in-progress Train (1 Epoch) at the next batch
  // boundary. Cleared at the start of each epoch run.
  let abortEpoch = $state(false);
  // Same pattern, for Train Continuously.
  let continuousRunning = $state(false);
  let abortContinuous = $state(false);

  // Number of batches in one "epoch" — derived from the user-set
  // samples-per-symbol target and the current batch size. Mirrors the
  // formula used in App.svelte for the Training tab subtitle.
  let batchesPerEpoch = $derived.by(() => {
    const classes = training.numClasses;
    const bs = architecture.hyperparameters.batch_size;
    if (classes <= 0 || bs <= 0) return 0;
    return Math.max(
      1,
      Math.ceil((classes * training.samplesPerSymbolPerEpoch) / bs)
    );
  });
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
    // Fetch initial state + checkpoint list. Batch loading is handled
    // by the $effect below — onMount used to call loadBatch() too, but
    // that caused a fresh batch to stream every time the user came back
    // to the Training tab even though the existing batch was still
    // valid. Now the effect only fetches when the config actually
    // changed (or the global batch is empty).
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
    } catch (e) {
      statusMsg = `init failed: ${(e as Error).message}`;
    }
  });

  // Decide whether to (re)load the batch. Triggered when batch_size
  // changes, synthesis becomes available, or any synthesis setting
  // changes — but on the *first* run after this component mounts, we
  // skip the load if the global batch is already populated. That means
  // navigating away from the Training tab and coming back doesn't burn
  // a fresh batch when the existing one is still valid; an actual
  // config change does invalidate it via the per-instance signature
  // diff. (See state.svelte.ts: training.batch lives in global state,
  // so it survives unmount/remount of TrainingTab.)
  let lastBatchConfigSig: string | null = null;
  $effect(() => {
    if (!synthesis.loaded) return;
    const sig = JSON.stringify({
      batch_size: architecture.hyperparameters.batch_size,
      cats: synthesis.selectedCategories,
      fonts: synthesis.fontUsage,
      aug: synthesis.augmentation,
    });
    const isFirstRun = lastBatchConfigSig === null;
    const configChanged = !isFirstRun && sig !== lastBatchConfigSig;
    lastBatchConfigSig = sig;
    if (isFirstRun && training.batch.length > 0) {
      // Coming back from another tab with a still-valid batch — keep
      // it on screen instead of regenerating.
      if (training.hasSession && training.predictions.length === 0) {
        void refreshPredictions();
      }
      return;
    }
    if (!isFirstRun && !configChanged) return;
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
    training.batchVerdict = new Array(samples.length).fill(null);
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

  // ── loss curves ────────────────────────────────────────────────────────

  // Cap each series so a long Train Continuously run can't grow them
  // unbounded; the most recent N points are what's interesting anyway.
  const MAX_LOSS_HISTORY = 2000;

  function recordTrainLoss(step: number, loss: number) {
    training.lossHistory.push({ step, loss });
    if (training.lossHistory.length > MAX_LOSS_HISTORY) {
      training.lossHistory = training.lossHistory.slice(-MAX_LOSS_HISTORY);
    }
  }

  // Run validation every Nth step using the held-out fonts. Skips
  // silently when no val fonts are configured (so curve just doesn't
  // populate) or when val labels fall outside the session's class set
  // (e.g., user changed categories without reinit).
  async function maybeRunValidation(step: number) {
    if (!training.hasSession) return;
    if (training.validateEveryN <= 0 || step <= 0) return;
    if (step % training.validateEveryN !== 0) return;
    const valFonts = synthesis.fonts.filter(
      (f) => synthesis.fontUsage[f.family] === 'val'
    );
    if (valFonts.length === 0) return;
    const cfg = {
      ...buildSampleRequest(architecture.hyperparameters.batch_size),
      split: 'val' as const,
    };
    try {
      const samples: SynthesisSample[] = [];
      for await (const s of streamSamples(cfg)) {
        samples.push(s);
      }
      if (samples.length === 0) return;
      const r = await api.evalBatch(
        samples.map((s) => s.png_b64),
        samples.map((s) => s.label)
      );
      training.valLossHistory.push({ step, loss: r.loss });
      if (training.valLossHistory.length > MAX_LOSS_HISTORY) {
        training.valLossHistory = training.valLossHistory.slice(
          -MAX_LOSS_HISTORY
        );
      }
    } catch (e) {
      // Don't surface val errors to the status footer — they shouldn't
      // disrupt training. Just drop this point and move on.
      console.warn('val skipped:', e);
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
      training.lastLoss = null;
      training.lastAccuracy = null;
      training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
      training.lossHistory = [];
      training.valLossHistory = [];
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
      // Reset verdicts for this animation pass — they fill in as we go and
      // persist on-screen until the next batch loads.
      training.batchVerdict = new Array(total).fill(null);
      // Reset the chart counts so the user sees them grow from zero
      // alongside the per-image sweep.
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
      // Apply the actual gradient step on the same batch.
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
      statusMsg =
        `step ${result.step} · loss ${result.loss.toFixed(4)} ` +
        `· acc ${(result.accuracy * 100).toFixed(1)}%`;
      // Load a new batch + new predictions for the next round.
      await loadBatch();
    } catch (e) {
      statusMsg = `train failed: ${(e as Error).message}`;
    } finally {
      training.animating = false;
      busy = false;
    }
  }

  // Train for one epoch == batchesPerEpoch sequential train_batch calls,
  // each on a fresh batch. Skips the per-image animation that
  // trainOneBatch does — at this scale (often 30+ batches) the animation
  // would take minutes. Status message + tab subtitle update live.
  async function trainOneEpoch() {
    if (!training.hasSession || batchesPerEpoch <= 0) return;
    if (training.batch.length === 0) {
      statusMsg = 'no batch loaded — check Data Synthesis';
      return;
    }
    busy = true;
    epochRunning = true;
    abortEpoch = false;
    const total = batchesPerEpoch;
    statusMsg = `training (1 epoch = ${total} batches)…`;
    try {
      for (let i = 0; i < total; i++) {
        if (abortEpoch) {
          statusMsg = `training epoch interrupted (batch ${i}/${total}) · step ${training.step}`;
          return;
        }
        if (training.batch.length === 0) {
          statusMsg = `epoch stopped: failed to load batch ${i + 1}/${total}`;
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
        statusMsg =
          `training epoch (batch ${i + 1}/${total}) · step ${result.step} ` +
          `· loss ${result.loss.toFixed(4)} ` +
          `· acc ${(result.accuracy * 100).toFixed(1)}%`;
        // Update the persistent batch chart with this step's verdicts
        // before the next loadBatch wipes the predictions.
        snapshotBatchChartCounts();
        // Stream a new batch for the next iteration (and to refresh the
        // grid when the user is watching).
        await loadBatch();
      }
      statusMsg =
        `epoch complete · step ${training.step} ` +
        (training.lastLoss !== null
          ? `· loss ${training.lastLoss.toFixed(4)} `
          : '') +
        (training.lastAccuracy !== null
          ? `· acc ${(training.lastAccuracy * 100).toFixed(1)}%`
          : '');
    } catch (e) {
      statusMsg = `epoch train failed: ${(e as Error).message}`;
    } finally {
      abortEpoch = false;
      epochRunning = false;
      busy = false;
    }
  }

  function stopEpoch() {
    abortEpoch = true;
  }

  // Same gradient step as trainOneBatch but without the per-image
  // highlight + prediction sweep — just train, update status, advance.
  async function trainOneBatchFast() {
    if (training.batch.length === 0 || !training.hasSession) return;
    busy = true;
    statusMsg = 'training (1 batch)…';
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
      statusMsg =
        `step ${result.step} · loss ${result.loss.toFixed(4)} ` +
        `· acc ${(result.accuracy * 100).toFixed(1)}%`;
      // Snapshot pre-step prediction counts BEFORE loadBatch swaps the
      // predictions array out from under us.
      snapshotBatchChartCounts();
      await loadBatch();
    } catch (e) {
      statusMsg = `train failed: ${(e as Error).message}`;
    } finally {
      busy = false;
    }
  }

  // Open-ended training loop — keeps calling train_batch until the user
  // hits Stop. Same shape as trainOneEpoch but without the batch cap;
  // status reports running totals (step + derived epoch count) instead of
  // progress within an epoch.
  async function trainContinuously() {
    if (!training.hasSession) return;
    if (training.batch.length === 0) {
      statusMsg = 'no batch loaded — check Data Synthesis';
      return;
    }
    busy = true;
    continuousRunning = true;
    abortContinuous = false;
    statusMsg = 'training continuously…';
    try {
      while (!abortContinuous) {
        if (training.batch.length === 0) {
          statusMsg = 'continuous stopped: failed to load batch';
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
        statusMsg =
          `continuous · step ${result.step} ` +
          `(${epochsDone} epoch${epochsDone === 1 ? '' : 's'}) ` +
          `· loss ${result.loss.toFixed(4)} ` +
          `· acc ${(result.accuracy * 100).toFixed(1)}%`;
        snapshotBatchChartCounts();
        await loadBatch();
      }
      statusMsg =
        `stopped · step ${training.step} ` +
        (training.lastLoss !== null
          ? `· loss ${training.lastLoss.toFixed(4)} `
          : '') +
        (training.lastAccuracy !== null
          ? `· acc ${(training.lastAccuracy * 100).toFixed(1)}%`
          : '');
    } catch (e) {
      statusMsg = `continuous train failed: ${(e as Error).message}`;
    } finally {
      abortContinuous = false;
      continuousRunning = false;
      busy = false;
    }
  }

  function stopContinuous() {
    abortContinuous = true;
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
      training.lastLoss = null;
      training.lastAccuracy = null;
      training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
      training.lossHistory = [];
      training.valLossHistory = [];
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
    // Lock in a verdict for this sample — same green/red feedback the
    // Train 1 Batch (Fun) sweep uses, so manual clicks classify too.
    recordVerdict(i);
  }

  // Compute correct/incorrect for sample `i` from the cached prediction
  // and write it into batchVerdict. No-op if prediction or sample missing.
  function recordVerdict(i: number) {
    const probs = training.predictions[i];
    const sample = training.batch[i];
    if (!probs || !sample) return;
    let bestIdx = 0;
    for (let k = 1; k < probs.length; k++) {
      if (probs[k] > probs[bestIdx]) bestIdx = k;
    }
    const predicted = selectedClasses[bestIdx];
    training.batchVerdict[i] =
      predicted === sample.label ? 'correct' : 'incorrect';
  }

  // Tally pre-step predictions for the current batch into batchChartCounts.
  // Used by Fast / Epoch / Continuous, which don't run the per-image sweep
  // and so need to snapshot the chart all at once. Must be called BEFORE
  // loadBatch — once that runs, training.predictions is replaced.
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
      if (selectedClasses[bestIdx] === sample.label) correct++;
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
      onTrainBatchFun={trainOneBatch}
      onTrainBatchFast={trainOneBatchFast}
      onTrainEpoch={trainOneEpoch}
      onStopEpoch={stopEpoch}
      onTrainContinuously={trainContinuously}
      onStopContinuous={stopContinuous}
      onSaveCheckpoint={saveCheckpoint}
      onLoadCheckpoint={loadCheckpoint}
      canTrain={training.hasSession}
      {canReinit}
      {reinitBlockedReason}
      {busy}
      {batchesPerEpoch}
      {epochRunning}
      {continuousRunning}
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
        {statusMsg}
      </footer>
    </div>

    <div class="w-px shrink-0 bg-[var(--color-border)]"></div>

    <div class="w-96 shrink-0 min-h-0 flex flex-col">
      <!-- Prediction section: per-sample probs + per-batch correctness -->
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
        <PredictionBar classes={selectedClasses} />
        <div class="w-px shrink-0 bg-[var(--color-border)]"></div>
        <div class="w-24 shrink-0 min-h-0 flex flex-col">
          <BatchChart />
        </div>
      </div>

      <!-- Loss section: line charts of training and validation loss
           over training steps. Validation runs every Nth step against
           a freshly-rendered held-out batch (skipped silently if no
           val fonts are configured). -->
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

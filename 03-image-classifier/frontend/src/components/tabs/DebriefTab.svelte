<script lang="ts">
  import {
    architecture,
    dataset,
    training,
    ui,
    INPUT_SHAPE,
    isDatasetReady,
    type TabId,
  } from '../../state.svelte';
  import {
    computeArchitecture,
    formatCount,
  } from './architecture/computeArchitecture';

  let datasetReady = $derived(isDatasetReady());

  let totalParams = $derived.by(() => {
    if (architecture.preset === 'resnet18') return training.paramCount || 11_181_642;
    const numClasses = dataset.classes.length || 10;
    return computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
      .totalParams;
  });

  let batchesPerEpoch = $derived.by(() => {
    const totalTrain = dataset.status?.num_train ?? 0;
    const bs = architecture.hyperparameters.batch_size;
    if (totalTrain <= 0 || bs <= 0) return 0;
    return Math.max(1, Math.ceil(totalTrain / bs));
  });

  let epochsDone = $derived.by(() => {
    if (!training.hasSession || batchesPerEpoch <= 0) return 0;
    return Math.floor(training.step / batchesPerEpoch);
  });

  let batchInEpoch = $derived.by(() => {
    if (!training.hasSession || batchesPerEpoch <= 0) return 0;
    return training.step % batchesPerEpoch;
  });

  let augmentationSummary = $derived.by(() => {
    const parts: string[] = [];
    if (dataset.augmentation.random_crop) parts.push('Random crop');
    if (dataset.augmentation.flip) parts.push('Horizontal flip');
    if (dataset.augmentation.jitter > 0)
      parts.push(`Color jitter (${dataset.augmentation.jitter.toFixed(2)})`);
    return parts.length > 0 ? parts.join(', ') : 'None';
  });

  let architectureLabel = $derived.by(() => {
    if (architecture.preset === 'lenet5') return 'LeNet-5 (1998)';
    if (architecture.preset === 'alexnet') return 'AlexNet (2012)';
    if (architecture.preset === 'resnet18') return 'ResNet-18 (2015)';
    if (architecture.layers.length > 0) return 'Custom';
    return null;
  });

  // Imagenette is a fixed-size, fixed-difficulty task — no need to scale
  // accuracy thresholds with class count like Math Symbols did. ~70% is
  // a respectable result for a from-scratch model in a few minutes;
  // 80%+ is excellent.
  const DECENT = 0.55;
  const GREAT = 0.78;

  type Status =
    | 'loading'
    | 'no-dataset'
    | 'no-architecture'
    | 'no-training'
    | 'just-loaded'
    | 'early-training'
    | 'progressing'
    | 'almost-there'
    | 'fully-trained';

  let status = $derived.by<Status>(() => {
    if (!dataset.loaded) return 'loading';
    if (!datasetReady) return 'no-dataset';
    if (
      architecture.preset !== 'resnet18' &&
      architecture.layers.length === 0
    )
      return 'no-architecture';
    if (!training.hasSession || training.step === 0) return 'no-training';
    if (training.lastAccuracy === null) return 'just-loaded';
    const acc = training.lastAccuracy;
    if (acc >= GREAT) return 'fully-trained';
    if (acc >= DECENT) return 'almost-there';
    if (acc >= DECENT * 0.6) return 'progressing';
    return 'early-training';
  });

  function goTo(tab: TabId) {
    ui.activeTab = tab;
  }

  function formatPct(x: number | null): string {
    if (x === null) return '—';
    return `${(x * 100).toFixed(1)}%`;
  }
</script>

<div class="h-full overflow-auto">
  <div class="max-w-3xl mx-auto px-6 py-10 flex flex-col gap-6">
    <h1 class="text-2xl font-bold text-[var(--color-heading)] tracking-tight">
      Debrief
    </h1>

    <p class="text-base text-[var(--color-text)] leading-relaxed">
      Here's where you stand on building and training your image classifier —
      the data, the model, and the progress so far.
    </p>

    <div
      class="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]/40
             divide-y divide-[var(--color-border)]"
    >
      <!-- Dataset -->
      <section class="p-4 flex flex-col gap-2">
        <h2 class="text-sm font-semibold text-[var(--color-heading)]">
          Data
        </h2>
        {#if !dataset.loaded}
          <p class="text-sm text-[var(--color-muted)]">Loading…</p>
        {:else if !datasetReady}
          <p class="text-sm text-[var(--color-muted)]">Not yet downloaded.</p>
        {:else}
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Dataset</dt>
            <dd class="text-[var(--color-text)]">Imagenette ({dataset.classes.length} classes)</dd>
            <dt class="text-[var(--color-muted)]">Train images</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {(dataset.status?.num_train ?? 0).toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">Val images</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {(dataset.status?.num_val ?? 0).toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">Augmentation</dt>
            <dd class="text-[var(--color-text)]">{augmentationSummary}</dd>
          </dl>
        {/if}
      </section>

      <!-- Architecture -->
      <section class="p-4 flex flex-col gap-2">
        <h2 class="text-sm font-semibold text-[var(--color-heading)]">
          Network Architecture
        </h2>
        {#if architectureLabel === null}
          <p class="text-sm text-[var(--color-muted)]">Not yet defined.</p>
        {:else}
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Preset</dt>
            <dd class="text-[var(--color-text)]">{architectureLabel}</dd>
            <dt class="text-[var(--color-muted)]">Parameters</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {formatCount(totalParams)} ({totalParams.toLocaleString()})
            </dd>
            <dt class="text-[var(--color-muted)]">Learning rate</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {architecture.hyperparameters.lr}
            </dd>
            <dt class="text-[var(--color-muted)]">Batch size</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {architecture.hyperparameters.batch_size}
            </dd>
            <dt class="text-[var(--color-muted)]">Optimizer</dt>
            <dd class="text-[var(--color-text)] uppercase">
              {architecture.hyperparameters.optimizer}
            </dd>
          </dl>
        {/if}
      </section>

      <!-- Training Progress -->
      <section class="p-4 flex flex-col gap-2">
        <h2 class="text-sm font-semibold text-[var(--color-heading)]">
          Training Progress
        </h2>
        {#if !training.hasSession}
          <p class="text-sm text-[var(--color-muted)]">
            No training session yet.
          </p>
        {:else}
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Steps trained</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {training.step.toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">Epochs</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {epochsDone.toLocaleString()}
              {#if batchInEpoch > 0}
                <span class="text-[var(--color-muted)]">
                  + {batchInEpoch} batches
                </span>
              {/if}
            </dd>
            <dt class="text-[var(--color-muted)]">Last batch loss</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {training.lastLoss === null
                ? '—'
                : training.lastLoss.toFixed(4)}
            </dd>
            <dt class="text-[var(--color-muted)]">Last batch accuracy</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {formatPct(training.lastAccuracy)}
            </dd>
          </dl>
        {/if}
      </section>
    </div>

    <!-- Contextual message -->
    <div class="text-base text-[var(--color-text)] leading-relaxed">
      {#if status === 'loading'}
        <p>Loading the dataset configuration…</p>
      {:else if status === 'no-dataset'}
        <p>
          The Imagenette dataset hasn't been downloaded yet. Head over to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('data')}>Data Acquisition</button>
          and click Download Imagenette — it's about 88 MB and takes a few
          seconds on a fast connection.
        </p>
      {:else if status === 'no-architecture'}
        <p>
          The dataset is downloaded — 10 classes, about
          {(dataset.status?.num_train ?? 0).toLocaleString()} training images.
          Next stop:
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('architecture')}>Model Architecture</button>
          — pick one of the three preset architectures (LeNet-5, AlexNet,
          ResNet-18) or design your own.
        </p>
      {:else if status === 'no-training'}
        <p>
          You have a {architectureLabel} network with
          <strong>{formatCount(totalParams)} parameters</strong>
          ready to learn 10 Imagenette classes. Time to train it! Head to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          — click <em>Initialize Model</em>, then <em>Train 1 Batch (Fun)</em>
          for the first run.
        </p>
      {:else if status === 'just-loaded'}
        <p>
          Your model has been trained for
          <strong>{training.step.toLocaleString()} steps</strong>
          previously. Run a few batches in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          to see how it's performing right now.
        </p>
      {:else if status === 'early-training'}
        <p>
          You've started training — accuracy is around
          <strong>{formatPct(training.lastAccuracy)}</strong> after
          {training.step.toLocaleString()} step{training.step === 1 ? '' : 's'}.
          That's normal at this stage. Head back to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and use <em>Train Continuously</em> — give the model a few thousand
          steps and the loss will start to come down properly.
        </p>
      {:else if status === 'progressing'}
        <p>
          You're on the right track — accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong> after
          {training.step.toLocaleString()} steps. The model is learning
          real visual features now. Keep going in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and you'll see it climb steadily.
        </p>
      {:else if status === 'almost-there'}
        <p>
          Coming along nicely! Accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong> — well past the
          random-guess floor. A few more epochs in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and you'll be there. You can also try the model on real images in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('inference')}>Inference</button>
          while it trains.
        </p>
      {:else if status === 'fully-trained'}
        <p>
          🎉 Congratulations! Your image classifier is performing well —
          accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong>
          on a {architectureLabel} model. Try it out in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('inference')}>Inference</button>
          — upload a photo of one of the 10 Imagenette categories and see
          how it does. And don't forget to save a checkpoint in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          if you haven't already.
        </p>
      {/if}
    </div>
  </div>
</div>

<style>
  .link-inline {
    color: var(--color-accent);
    text-decoration: underline;
    text-underline-offset: 2px;
    background: none;
    border: 0;
    padding: 0;
    font: inherit;
    cursor: pointer;
  }
  .link-inline:hover {
    text-decoration-thickness: 2px;
  }
</style>

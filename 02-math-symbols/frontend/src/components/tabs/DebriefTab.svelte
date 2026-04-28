<script lang="ts">
  import {
    architecture,
    synthesis,
    training,
    ui,
    INPUT_SHAPE,
    type TabId,
  } from '../../state.svelte';
  import {
    computeArchitecture,
    formatCount,
  } from './architecture/computeArchitecture';

  // ── Derived metrics ────────────────────────────────────────────────────

  let symbolCount = $derived.by(() => {
    if (!synthesis.loaded) return 0;
    let n = 0;
    for (const c of synthesis.categories) {
      if (synthesis.selectedCategories[c.id]) n += c.count;
    }
    return n;
  });

  let trainFontCount = $derived.by(() => {
    if (!synthesis.loaded) return 0;
    let n = 0;
    for (const f of synthesis.fonts) {
      if (synthesis.fontUsage[f.family] === 'train') n++;
    }
    return n;
  });

  let valFontCount = $derived.by(() => {
    if (!synthesis.loaded) return 0;
    let n = 0;
    for (const f of synthesis.fonts) {
      if (synthesis.fontUsage[f.family] === 'val') n++;
    }
    return n;
  });

  let augmentationSummary = $derived.by(() => {
    const parts: string[] = [];
    if (synthesis.augmentation.noise.enabled) {
      parts.push(`Noise (${synthesis.augmentation.noise.max_level}%)`);
    }
    if (synthesis.augmentation.skew.enabled) parts.push('Skew');
    return parts.length > 0 ? parts.join(', ') : 'None';
  });

  let totalParams = $derived.by(() => {
    const numClasses = symbolCount || 10;
    return computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
      .totalParams;
  });

  let batchesPerEpoch = $derived.by(() => {
    const classes = training.numClasses || symbolCount;
    const bs = architecture.hyperparameters.batch_size;
    if (classes <= 0 || bs <= 0) return 0;
    return Math.max(
      1,
      Math.ceil((classes * training.samplesPerSymbolPerEpoch) / bs)
    );
  });

  let epochsDone = $derived.by(() => {
    if (!training.hasSession || batchesPerEpoch <= 0) return 0;
    return Math.floor(training.step / batchesPerEpoch);
  });

  let batchInEpoch = $derived.by(() => {
    if (!training.hasSession || batchesPerEpoch <= 0) return 0;
    return training.step % batchesPerEpoch;
  });

  // ── Difficulty + accuracy thresholds ───────────────────────────────────
  //
  // The "great accuracy" bar scales with how hard the pipeline is. Tiny
  // class sets (digits-only) are easy to push over 95%; a few hundred
  // glyphs across many fonts is genuinely hard, and 75% there is already
  // an achievement.

  type Difficulty = 'easy' | 'medium' | 'advanced';

  let difficulty = $derived.by<Difficulty>(() => {
    if (symbolCount < 30) return 'easy';
    if (symbolCount <= 150) return 'medium';
    return 'advanced';
  });

  let difficultyDescriptor = $derived.by(() => {
    switch (difficulty) {
      case 'easy':
        return 'easy';
      case 'medium':
        return 'moderately challenging';
      case 'advanced':
        return 'challenging';
    }
  });

  // (decent, great) accuracy thresholds in [0, 1].
  let thresholds = $derived.by<[number, number]>(() => {
    switch (difficulty) {
      case 'easy':
        return [0.85, 0.95];
      case 'medium':
        return [0.7, 0.88];
      case 'advanced':
        return [0.55, 0.78];
    }
  });

  // ── Status classification ──────────────────────────────────────────────

  type Status =
    | 'loading'
    | 'no-symbols'
    | 'no-architecture'
    | 'no-training'
    | 'just-loaded'
    | 'early-training'
    | 'progressing'
    | 'almost-there'
    | 'fully-trained';

  let status = $derived.by<Status>(() => {
    if (!synthesis.loaded) return 'loading';
    if (symbolCount === 0) return 'no-symbols';
    if (architecture.layers.length === 0) return 'no-architecture';
    if (!training.hasSession || training.step === 0) return 'no-training';
    if (training.lastAccuracy === null) return 'just-loaded';
    const [decent, great] = thresholds;
    const acc = training.lastAccuracy;
    if (acc >= great) return 'fully-trained';
    if (acc >= decent) return 'almost-there';
    if (acc >= decent * 0.6) return 'progressing';
    return 'early-training';
  });

  // ── Helpers ────────────────────────────────────────────────────────────

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
      Here's where you stand on building and training your math-symbol
      classifier — the data, the model, and the progress so far.
    </p>

    <!-- ── Summary box ──────────────────────────────────────────────── -->
    <div
      class="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]/40
             divide-y divide-[var(--color-border)]"
    >
      <!-- Data Synthesis -->
      <section class="p-4 flex flex-col gap-2">
        <h2 class="text-sm font-semibold text-[var(--color-heading)]">
          Data Synthesis
        </h2>
        {#if !synthesis.loaded}
          <p class="text-sm text-[var(--color-muted)]">Loading…</p>
        {:else}
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Symbols</dt>
            <dd class="text-[var(--color-text)]">
              {symbolCount.toLocaleString()}
            </dd>
            <dt class="text-[var(--color-muted)]">Training fonts</dt>
            <dd class="text-[var(--color-text)]">{trainFontCount}</dd>
            <dt class="text-[var(--color-muted)]">Validation fonts</dt>
            <dd class="text-[var(--color-text)]">{valFontCount}</dd>
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
        {#if architecture.layers.length === 0}
          <p class="text-sm text-[var(--color-muted)]">
            Not yet defined.
          </p>
        {:else}
          <dl class="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
            <dt class="text-[var(--color-muted)]">Layers</dt>
            <dd class="text-[var(--color-text)]">
              {architecture.layers.length}
            </dd>
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
            <dt class="text-[var(--color-muted)]">Classes</dt>
            <dd class="text-[var(--color-text)] font-mono">
              {training.numClasses.toLocaleString()}
            </dd>
          </dl>
        {/if}
      </section>
    </div>

    <!-- ── Contextual message ───────────────────────────────────────── -->
    <div class="text-base text-[var(--color-text)] leading-relaxed">
      {#if status === 'loading'}
        <p>Loading the data synthesis configuration…</p>
      {:else if status === 'no-symbols'}
        <p>
          Your data synthesis has no symbols selected yet. Head over to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('data')}>Data Synthesis</button>
          to choose which categories of symbols you want the model to
          recognize.
        </p>
      {:else if status === 'no-architecture'}
        <p>
          Your data synthesis pipeline has
          <strong>{symbolCount.toLocaleString()} symbols</strong>
          across {trainFontCount} training font{trainFontCount === 1
            ? ''
            : 's'}. This will be a {difficultyDescriptor} training task!
          Next stop:
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('architecture')}>Model Architecture</button>
          — design (or auto-suggest) the neural network you want to train.
        </p>
      {:else if status === 'no-training'}
        <p>
          You have a network with
          <strong>{formatCount(totalParams)} parameters</strong>
          across {architecture.layers.length} layer{architecture.layers
            .length === 1
            ? ''
            : 's'}, ready to learn
          <strong>{symbolCount.toLocaleString()} symbols</strong>. Time to
          train it! Head to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          — click <em>Initialize Model</em>, then
          <em>Train 1 Batch (Fun)</em> for the first run.
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
          {training.step.toLocaleString()} step{training.step === 1
            ? ''
            : 's'}. That's normal at this stage. Head back to
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and use <em>Train Continuously</em> — give the model a few
          thousand steps and the loss will start to come down properly.
        </p>
      {:else if status === 'progressing'}
        <p>
          You're on the right track — accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong> after
          {training.step.toLocaleString()} steps. The model is learning
          real structure now. Keep going in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and you'll see it climb steadily.
        </p>
      {:else if status === 'almost-there'}
        <p>
          Coming along nicely! Accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong> — well past
          the noise floor for this {difficultyDescriptor} task. A few more
          epochs in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('training')}>Training</button>
          and you'll be there. You can also try out the model on real
          input in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('inference')}>Inference</button>
          while it trains.
        </p>
      {:else if status === 'fully-trained'}
        <p>
          🎉 Congratulations! Your math symbol recognition model is fully
          trained — accuracy is at
          <strong>{formatPct(training.lastAccuracy)}</strong> on a
          {difficultyDescriptor} pipeline of
          {symbolCount.toLocaleString()} symbols across {trainFontCount}
          training font{trainFontCount === 1 ? '' : 's'}. Try it out in
          <button
            type="button"
            class="link-inline"
            onclick={() => goTo('inference')}>Inference</button>
          — type some math and watch it classify each glyph. And don't
          forget to save a checkpoint in
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

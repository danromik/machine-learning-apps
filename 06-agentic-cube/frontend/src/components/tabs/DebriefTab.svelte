<script lang="ts">
  import {
    ui,
    cube,
    algorithm,
    algorithmInfo,
    training,
    report,
    chat,
    persistChatPrefs,
  } from '../../state.svelte';
  import Markdown from '../Markdown.svelte';

  // Best solve-rate achieved at the deepest evaluated depth.
  let deepestSolved = $derived.by(() => {
    const rows = Object.entries(training.solveRateByK)
      .map(([k, v]) => ({ k: Number(k), rate: v }))
      .sort((a, b) => b.k - a.k);
    return rows.length ? rows[0] : null;
  });

  function openCoach() {
    chat.visible = true;
    persistChatPrefs();
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-4xl mx-auto flex flex-col gap-5">
    <div>
      <h2 class="text-lg font-bold text-[var(--color-heading)]">Debrief</h2>
      <p class="text-sm text-[var(--color-muted)] mt-1">
        The RL Coach's written summary of the training run.
      </p>
    </div>

    <!-- Fact cards -->
    <div class="grid gap-3" style="grid-template-columns: repeat(2, 1fr)">
      <div class="card p-4">
        <div class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)] mb-2">
          Setup
        </div>
        <dl class="text-sm flex flex-col gap-1 text-[var(--color-text)]">
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Cube</dt><dd>{cube.size}×{cube.size}×{cube.size}</dd></div>
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Algorithm</dt><dd>{algorithmInfo()?.label ?? algorithm.algo}</dd></div>
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Curriculum</dt><dd>k {cube.curriculum.startK}–{cube.curriculum.maxK}</dd></div>
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Learning rate</dt><dd class="font-mono">{algorithm.hyperparameters.lr ?? '—'}</dd></div>
        </dl>
      </div>
      <div class="card p-4">
        <div class="text-xs font-semibold uppercase tracking-wide text-[var(--color-heading)] mb-2">
          Progress
        </div>
        <dl class="text-sm flex flex-col gap-1 text-[var(--color-text)]">
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Iterations</dt><dd class="font-mono">{training.iteration}</dd></div>
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Current depth k</dt><dd class="font-mono">{training.currentK}</dd></div>
          <div class="flex justify-between"><dt class="text-[var(--color-muted)]">Parameters</dt><dd class="font-mono">{training.paramCount ? training.paramCount.toLocaleString() : '—'}</dd></div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Best deep solve</dt>
            <dd class="font-mono">
              {deepestSolved ? `${(deepestSolved.rate * 100).toFixed(0)}% @ k=${deepestSolved.k}` : '—'}
            </dd>
          </div>
        </dl>
      </div>
    </div>

    <!-- The Coach-written report -->
    <div class="card p-5">
      {#if report.markdown}
        {#if !report.final}
          <div class="text-[11px] uppercase tracking-wide text-[var(--color-muted)] mb-2">
            Live report (not yet finalized — click Finish Training on the Progress
            Report tab for a full debrief)
          </div>
        {/if}
        <Markdown source={report.markdown} />
      {:else}
        <div class="text-sm text-[var(--color-muted)]">
          No debrief yet. Run training, then click
          <button class="underline" onclick={() => (ui.activeTab = 'progress-report')}
            >Finish Training</button
          >
          on the Progress Report tab and the RL Coach will write a full debrief here.
          You can also <button class="underline" onclick={openCoach}>ask the RL Coach</button>
          directly.
        </div>
      {/if}
    </div>
  </div>
</div>

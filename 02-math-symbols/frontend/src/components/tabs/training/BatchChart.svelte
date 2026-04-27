<script lang="ts">
  import { training } from '../../../state.svelte';

  // Counts come from training.batchChartCounts, which is updated:
  //   - live during Train 1 Batch (Fun)'s sweep
  //   - once at the end of Train 1 Batch (Fast)
  //   - after every batch step in Train 1 Epoch / Train Continuously
  // and persists across loadBatch so the last run's tally stays on
  // screen until another run replaces it.
  let counts = $derived(training.batchChartCounts);
  // Bar heights scale by the snapshot's batch size — not the live
  // training.batch.length — so a 32-batch snapshot stays full-height
  // even if the user later bumps batch_size.
  let total = $derived(Math.max(1, counts.total));
</script>

<div class="flex-1 min-w-0 min-h-0 flex flex-col p-3 gap-2">
  <header class="text-xs">
    <span class="font-semibold text-[var(--color-heading)]">Batch</span>
  </header>

  <div class="flex-1 min-h-0 flex items-end justify-around gap-2 pb-1">
    <div class="flex flex-col items-center justify-end gap-1 h-full">
      <span class="text-[11px] font-mono tabular-nums text-[var(--color-success)]">
        {counts.correct}
      </span>
      <div
        class="batch-chart-bar"
        style="
          height: {(counts.correct / total) * 100}%;
          background: var(--color-success);
        "
        title="{counts.correct} correctly predicted"
      ></div>
      <span class="text-[10px] text-[var(--color-muted)] font-mono">✓</span>
    </div>
    <div class="flex flex-col items-center justify-end gap-1 h-full">
      <span class="text-[11px] font-mono tabular-nums text-[var(--color-danger)]">
        {counts.incorrect}
      </span>
      <div
        class="batch-chart-bar"
        style="
          height: {(counts.incorrect / total) * 100}%;
          background: var(--color-danger);
        "
        title="{counts.incorrect} incorrectly predicted"
      ></div>
      <span class="text-[10px] text-[var(--color-muted)] font-mono">✗</span>
    </div>
  </div>

  {#if counts.total > 0}
    <footer class="text-[10px] text-[var(--color-muted)] font-mono text-center">
      {counts.correct + counts.incorrect} / {counts.total} classified
    </footer>
  {/if}
</div>

<style>
  .batch-chart-bar {
    width: 1.75rem;
    min-height: 2px;
    border-radius: 2px 2px 0 0;
    transition: height 120ms ease-out;
  }
</style>

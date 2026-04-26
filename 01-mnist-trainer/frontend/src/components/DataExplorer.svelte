<script lang="ts">
  import { explorer, ui } from '../state.svelte';
  import { api, type Sample } from '../api';

  const CELL_W = 44; // px horizontal pitch per cell
  const CELL_H = 56; // px vertical pitch per cell (image + label)
  const IMG = 40;
  const CHUNK = 500;
  const OVERSCAN_ROWS = 2;

  let container: HTMLDivElement | undefined;
  let containerW = $state(0);
  let containerH = $state(0);
  let scrollTop = $state(0);

  let total = $state(0);
  let cache = $state<Record<number, Sample>>({});
  const inflight = new Set<number>();

  const cols = $derived(Math.max(1, Math.floor(containerW / CELL_W)));
  const rows = $derived(Math.ceil(total / cols));
  const totalH = $derived(rows * CELL_H);

  const startRow = $derived(
    Math.max(0, Math.floor(scrollTop / CELL_H) - OVERSCAN_ROWS)
  );
  const endRow = $derived(
    Math.min(rows, Math.ceil((scrollTop + containerH) / CELL_H) + OVERSCAN_ROWS)
  );
  const startIdx = $derived(startRow * cols);
  const endIdx = $derived(Math.min(total, endRow * cols));

  type Cell = { idx: number; row: number; col: number; sample: Sample | null };
  const visible = $derived.by<Cell[]>(() => {
    const out: Cell[] = [];
    for (let i = startIdx; i < endIdx; i++) {
      out.push({
        idx: i,
        row: Math.floor(i / cols),
        col: i % cols,
        sample: cache[i] ?? null,
      });
    }
    return out;
  });

  async function fetchChunk(chunkId: number) {
    if (inflight.has(chunkId)) return;
    inflight.add(chunkId);
    try {
      const offset = chunkId * CHUNK;
      const res = await api.sample(
        explorer.split,
        explorer.cls,
        explorer.order,
        CHUNK,
        offset
      );
      for (let i = 0; i < res.samples.length; i++) {
        cache[offset + i] = res.samples[i];
      }
      if (res.total !== total) total = res.total;
    } catch (e) {
      console.error('chunk fetch failed', e);
    } finally {
      inflight.delete(chunkId);
    }
  }

  function ensureWindow() {
    if (total === 0 || endIdx <= startIdx) return;
    const firstChunk = Math.floor(startIdx / CHUNK);
    const lastChunk = Math.floor((endIdx - 1) / CHUNK);
    for (let c = firstChunk; c <= lastChunk; c++) {
      const start = c * CHUNK;
      const end = Math.min(total, start + CHUNK);
      let needs = false;
      for (let i = start; i < end; i++) {
        if (!cache[i]) {
          needs = true;
          break;
        }
      }
      if (needs) fetchChunk(c);
    }
  }

  async function refreshAll() {
    explorer.loading = true;
    try {
      const cls = explorer.cls === 'all' ? null : Number(explorer.cls);
      const counts = await api.counts(cls);
      explorer.trainCount = counts.train;
      explorer.testCount = counts.test;
      total = explorer.split === 'train' ? counts.train : counts.test;
    } catch (e) {
      console.error('counts fetch failed', e);
    } finally {
      explorer.loading = false;
    }
  }

  function resetAndRefresh() {
    cache = {};
    inflight.clear();
    total = 0;
    if (container) container.scrollTop = 0;
    scrollTop = 0;
    refreshAll();
  }

  function onScroll() {
    if (!container) return;
    scrollTop = container.scrollTop;
  }

  // Refetch counts and clear cache whenever filter changes.
  $effect(() => {
    explorer.split;
    explorer.cls;
    explorer.order;
    resetAndRefresh();
  });

  // Whenever the visible window or total/cols change, request any missing chunks.
  $effect(() => {
    startIdx;
    endIdx;
    total;
    ensureWindow();
  });

  const CLASS_OPTIONS = ['all', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
</script>

<section class="card flex flex-col min-h-0 min-w-0 flex-1 overflow-hidden">
  <div class="flex items-center justify-between px-4 pt-3 pb-2 gap-3">
    <div class="flex items-center gap-3 min-w-0">
      <h2 class="text-sm font-semibold tracking-tight">Data Explorer</h2>
      <span class="text-xs text-[var(--color-muted)] truncate">
        Dataset: training -
        <span class="text-[var(--color-text)] font-mono">
          {explorer.trainCount.toLocaleString()}
        </span>
        items, testing -
        <span class="text-[var(--color-text)] font-mono">
          {explorer.testCount.toLocaleString()}
        </span>
        items
      </span>
    </div>
    <div class="flex items-center gap-2 shrink-0">
      <label class="text-xs text-[var(--color-muted)] flex items-center gap-1.5">
        Split
        <select class="input !py-1 !px-2" bind:value={explorer.split}>
          <option value="train">train</option>
          <option value="test">test</option>
        </select>
      </label>
      <label class="text-xs text-[var(--color-muted)] flex items-center gap-1.5">
        Class
        <select class="input !py-1 !px-2" bind:value={explorer.cls}>
          {#each CLASS_OPTIONS as c}
            <option value={c}>{c}</option>
          {/each}
        </select>
      </label>
      <label class="text-xs text-[var(--color-muted)] flex items-center gap-1.5">
        Sort
        <select class="input !py-1 !px-2" bind:value={explorer.order}>
          <option value="default">Default</option>
          <option value="digit">By digit</option>
        </select>
      </label>
    </div>
  </div>
  <div
    bind:this={container}
    bind:clientWidth={containerW}
    bind:clientHeight={containerH}
    onscroll={onScroll}
    class="flex-1 overflow-auto px-4 pb-3"
  >
    {#if total === 0 && explorer.loading}
      <div class="text-xs text-[var(--color-muted)] py-4">loading…</div>
    {:else}
      <div class="relative" style="height: {totalH}px">
        {#each visible as cell (cell.idx)}
          {#if cell.sample}
            {@const sample = cell.sample}
            <button
              type="button"
              title="Click to classify this digit"
              onclick={() =>
                (ui.loadImage = {
                  png_b64: sample.png_b64,
                  label: sample.label,
                  seq: (ui.loadImage?.seq ?? 0) + 1,
                })}
              class="absolute flex flex-col items-center bg-transparent border-0 p-0 cursor-pointer focus:outline-none group"
              style="left: {cell.col * CELL_W}px; top: {cell.row * CELL_H}px; width: {CELL_W}px; height: {CELL_H}px"
            >
              <img
                src="data:image/png;base64,{sample.png_b64}"
                alt={String(sample.label)}
                class="pixel-img rounded-sm bg-black ring-0 group-hover:ring-2 group-focus:ring-2 ring-[var(--color-accent)] transition-shadow"
                style="width: {IMG}px; height: {IMG}px"
              />
              <span class="text-[10px] text-[var(--color-muted)] font-mono leading-tight mt-0.5">
                {sample.label}
              </span>
            </button>
          {:else}
            <div
              class="absolute flex flex-col items-center"
              style="left: {cell.col * CELL_W}px; top: {cell.row * CELL_H}px; width: {CELL_W}px; height: {CELL_H}px"
            >
              <div
                class="rounded-sm bg-[var(--color-surface-2)]"
                style="width: {IMG}px; height: {IMG}px"
              ></div>
            </div>
          {/if}
        {/each}
      </div>
    {/if}
  </div>
</section>

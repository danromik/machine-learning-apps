<script lang="ts">
  // A small line chart: faint raw per-episode points + a bold rolling-mean
  // line. Pure SVG so it stays crisp and dependency-free.
  let {
    values,
    label = 'Score',
    height = 220,
    smoothWindow = 25,
    color = 'var(--color-accent)',
  }: {
    values: number[];
    label?: string;
    height?: number;
    smoothWindow?: number;
    color?: string;
  } = $props();

  const W = 600;
  const PAD_L = 36;
  const PAD_B = 20;
  const PAD_T = 10;
  const PAD_R = 8;

  function rollingMean(xs: number[], win: number): number[] {
    const out: number[] = [];
    let sum = 0;
    const q: number[] = [];
    for (const x of xs) {
      q.push(x);
      sum += x;
      if (q.length > win) sum -= q.shift()!;
      out.push(sum / q.length);
    }
    return out;
  }

  let smoothed = $derived(rollingMean(values, smoothWindow));
  let maxY = $derived(Math.max(1, ...values));
  let n = $derived(values.length);

  function px(i: number): number {
    if (n <= 1) return PAD_L;
    return PAD_L + (i / (n - 1)) * (W - PAD_L - PAD_R);
  }
  function py(v: number): number {
    return PAD_T + (1 - v / maxY) * (height - PAD_T - PAD_B);
  }

  let rawPath = $derived(
    values.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
  );
  let smoothPath = $derived(
    smoothed.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
  );
  let lastSmoothed = $derived(smoothed.length ? smoothed[smoothed.length - 1] : 0);
</script>

<div class="card p-3">
  <div class="flex items-baseline justify-between mb-1">
    <span class="text-xs font-semibold text-[var(--color-heading)]">{label}</span>
    <span class="text-xs text-[var(--color-muted)] font-mono">
      {n > 0 ? `avg ${lastSmoothed.toFixed(2)} · max ${maxY}` : '—'}
    </span>
  </div>
  {#if n === 0}
    <div
      class="flex items-center justify-center text-[var(--color-muted)] text-sm"
      style="height:{height}px"
    >
      No episodes yet
    </div>
  {:else}
    <svg viewBox="0 0 {W} {height}" class="w-full" style="height:{height}px">
      <!-- y gridlines at 0, mid, max -->
      {#each [0, 0.5, 1] as f}
        <line
          x1={PAD_L}
          x2={W - PAD_R}
          y1={py(maxY * f)}
          y2={py(maxY * f)}
          stroke="var(--color-border)"
          stroke-width="1"
          opacity="0.5"
        />
        <text
          x={PAD_L - 5}
          y={py(maxY * f) + 3}
          text-anchor="end"
          font-size="10"
          fill="var(--color-muted)"
        >
          {Math.round(maxY * f)}
        </text>
      {/each}
      <path d={rawPath} fill="none" stroke={color} stroke-width="1" opacity="0.25" />
      <path d={smoothPath} fill="none" stroke={color} stroke-width="2" />
    </svg>
  {/if}
</div>

<script lang="ts">
  // Simple line chart for a {step, loss} time series. The right panel
  // renders two of these side-by-side (Training, Validation), so the
  // component just takes its data + label + color and draws the line —
  // the surrounding section header lives in TrainingTab.svelte.

  let {
    data,
    color,
    label,
  }: {
    data: { step: number; loss: number }[];
    color: string;
    label: string;
  } = $props();

  let containerWidth = $state(0);
  let containerHeight = $state(0);

  // Auto-scale Y to the data's [0, max*1.05] so the line never clips at
  // the top, with a tiny floor so a degenerate single-point series
  // still renders something visible.
  let domain = $derived.by(() => {
    if (data.length === 0) {
      return { stepMin: 0, stepMax: 1, lossMax: 1 };
    }
    let stepMin = data[0].step;
    let stepMax = data[0].step;
    let lossMax = data[0].loss;
    for (const d of data) {
      if (d.step < stepMin) stepMin = d.step;
      if (d.step > stepMax) stepMax = d.step;
      if (d.loss > lossMax) lossMax = d.loss;
    }
    return {
      stepMin,
      stepMax: Math.max(stepMax, stepMin + 1),
      lossMax: Math.max(0.001, lossMax * 1.05),
    };
  });

  // Plot-area paddings — leave room for axis labels around the inner
  // line plot. The Y-axis label width is sized for a "0.00" loss value;
  // the X-axis label space is one line of 9-px text + tick. RIGHT_PAD
  // accounts for the rightmost x-axis label being center-anchored on
  // its tick — half the label width extends beyond plotW, so the
  // padding must be at least half a max-width label (e.g. "1.5k" or
  // "752" ≈ 18px wide → ~10px overhang).
  const LEFT_PAD = 32;
  const RIGHT_PAD = 14;
  const TOP_PAD = 4;
  const BOTTOM_PAD = 14;

  let plotW = $derived(
    Math.max(1, containerWidth - LEFT_PAD - RIGHT_PAD)
  );
  let plotH = $derived(
    Math.max(1, containerHeight - TOP_PAD - BOTTOM_PAD)
  );

  // Format a step number compactly for the X-axis label. Anything in
  // the thousands range gets a "k" suffix to keep labels narrow.
  function fmtStep(n: number): string {
    const v = Math.round(n);
    if (v >= 10000) return `${Math.round(v / 1000)}k`;
    if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return String(v);
  }

  // Format a loss value for the Y-axis. Uses 2 decimals up to 9.99,
  // then 1 decimal for larger values, then plain integer.
  function fmtLoss(v: number): string {
    if (v >= 100) return v.toFixed(0);
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  // Tick mark positions in pixel coordinates.
  let yTicks = $derived.by(() => {
    const { lossMax } = domain;
    const n = plotH < 50 ? 2 : 4;
    const ticks: { value: number; y: number }[] = [];
    for (let i = 0; i <= n; i++) {
      const value = (lossMax * i) / n;
      const y = TOP_PAD + plotH - (value / lossMax) * plotH;
      ticks.push({ value, y });
    }
    return ticks;
  });

  let xTicks = $derived.by(() => {
    const { stepMin, stepMax } = domain;
    const range = stepMax - stepMin;
    // Avoid duplicate labels for tiny ranges (e.g., single data point).
    const n = Math.min(4, Math.max(1, Math.round(range)));
    const ticks: { value: number; x: number }[] = [];
    for (let i = 0; i <= n; i++) {
      const value = stepMin + (range * i) / n;
      const x = LEFT_PAD + (i / n) * plotW;
      ticks.push({ value, x });
    }
    return ticks;
  });

  // Polyline points string. Computed in pixel coordinates against the
  // bound container size so the chart resizes responsively.
  let pointsStr = $derived.by(() => {
    if (data.length === 0 || plotW <= 0 || plotH <= 0) return '';
    const { stepMin, stepMax, lossMax } = domain;
    const stepRange = Math.max(1, stepMax - stepMin);
    const out: string[] = [];
    for (const d of data) {
      const x = LEFT_PAD + ((d.step - stepMin) / stepRange) * plotW;
      const y = TOP_PAD + plotH - (d.loss / lossMax) * plotH;
      out.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return out.join(' ');
  });

  let lastLoss = $derived(
    data.length > 0 ? data[data.length - 1].loss : null
  );
</script>

<div class="flex-1 min-w-0 min-h-0 flex flex-col p-3 gap-2">
  <header class="flex items-baseline justify-between text-xs">
    <span class="font-semibold text-[var(--color-heading)]">{label}</span>
    {#if lastLoss !== null}
      <span class="text-[var(--color-muted)] font-mono tabular-nums">
        {lastLoss.toFixed(3)} · {data.length} pts
      </span>
    {/if}
  </header>

  <div
    bind:clientWidth={containerWidth}
    bind:clientHeight={containerHeight}
    class="flex-1 min-h-0 min-w-0 relative"
  >
    {#if data.length === 0}
      <div
        class="absolute inset-0 flex items-center justify-center
               text-[10px] text-[var(--color-muted)] font-mono"
      >
        — no data yet —
      </div>
    {:else}
      <svg
        width={containerWidth}
        height={containerHeight}
        class="block"
        role="img"
        aria-label={`${label} over training steps`}
      >
        <!-- Faint horizontal gridlines + Y-axis tick labels -->
        {#each yTicks as t}
          <line
            x1={LEFT_PAD}
            y1={t.y}
            x2={LEFT_PAD + plotW}
            y2={t.y}
            stroke="var(--color-border)"
            stroke-opacity={t.value === 0 ? 0.6 : 0.25}
            stroke-width="1"
          />
          <text
            x={LEFT_PAD - 4}
            y={t.y + 3}
            text-anchor="end"
            font-size="9"
            fill="var(--color-muted)"
            style="font-family: var(--font-mono)"
          >{fmtLoss(t.value)}</text>
        {/each}

        <!-- X-axis tick marks + labels -->
        {#each xTicks as t}
          <line
            x1={t.x}
            y1={TOP_PAD + plotH}
            x2={t.x}
            y2={TOP_PAD + plotH + 3}
            stroke="var(--color-border)"
            stroke-width="1"
          />
          <text
            x={t.x}
            y={TOP_PAD + plotH + 11}
            text-anchor="middle"
            font-size="9"
            fill="var(--color-muted)"
            style="font-family: var(--font-mono)"
          >{fmtStep(t.value)}</text>
        {/each}

        <!-- Y-axis line -->
        <line
          x1={LEFT_PAD}
          y1={TOP_PAD}
          x2={LEFT_PAD}
          y2={TOP_PAD + plotH}
          stroke="var(--color-border)"
          stroke-width="1"
        />

        <!-- The data line -->
        <polyline
          points={pointsStr}
          fill="none"
          stroke={color}
          stroke-width="1.5"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
      </svg>
    {/if}
  </div>
</div>

<script lang="ts">
  import uPlot from 'uplot';
  import type { Options, AlignedData } from 'uplot';
  import { chartData, theme } from '../state.svelte';
  import { cssVar } from '../theme';

  let container: HTMLDivElement | undefined;
  let plot: uPlot | null = null;

  function makeOpts(width: number, height: number): Options {
    const axisStroke = cssVar('--color-muted') || '#9ca3af';
    const gridStroke = cssVar('--color-border') || '#2a2a33';
    const accent = cssVar('--color-accent') || '#60a5fa';
    return {
      width,
      height,
      padding: [12, 16, 4, 8],
      legend: { show: false },
      cursor: { drag: { x: false, y: false } },
      scales: { x: { time: false } },
      axes: [
        { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: gridStroke } },
        { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: gridStroke } },
      ],
      series: [
        {},
        { label: 'train_loss', stroke: accent, width: 1.5, points: { show: false } },
      ],
    };
  }

  function build() {
    if (!container) return;
    plot?.destroy();
    const xs = chartData.steps.length === 0 ? [0] : chartData.steps;
    const ys = chartData.losses.length === 0 ? [null] : chartData.losses;
    plot = new uPlot(
      makeOpts(container.clientWidth, container.clientHeight),
      [xs, ys] as AlignedData,
      container,
    );
  }

  $effect(() => {
    // React to theme changes (rebuild with new colors).
    theme.version;
    build();
    if (!container) return;
    const ro = new ResizeObserver(() => {
      if (!plot || !container) return;
      plot.setSize({ width: container.clientWidth, height: container.clientHeight });
    });
    ro.observe(container);
    return () => {
      ro.disconnect();
      plot?.destroy();
      plot = null;
    };
  });

  $effect(() => {
    if (!plot) return;
    const xs = chartData.steps.length === 0 ? [0] : chartData.steps;
    const ys = chartData.losses.length === 0 ? [null] : chartData.losses;
    plot.setData([xs, ys] as AlignedData);
  });
</script>

<div class="card flex flex-col min-h-0 flex-1 min-w-0 overflow-hidden">
  <div class="px-3 pt-2 text-xs font-semibold text-[var(--color-muted)]">
    Training loss (per step)
  </div>
  <div bind:this={container} class="flex-1 min-h-0"></div>
</div>

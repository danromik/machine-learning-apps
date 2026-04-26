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
    const danger = cssVar('--color-danger') || '#f87171';
    const success = cssVar('--color-success') || '#34d399';
    return {
      width,
      height,
      padding: [12, 8, 4, 8],
      legend: { show: true, live: false },
      cursor: { drag: { x: false, y: false } },
      scales: {
        x: { time: false },
        y: { auto: true },
        y2: { auto: false, range: [0, 1] },
      },
      axes: [
        { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: gridStroke } },
        { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: gridStroke } },
        {
          scale: 'y2',
          side: 1,
          stroke: axisStroke,
          grid: { show: false },
          ticks: { stroke: gridStroke },
          values: (_u, vals) => vals.map((v) => v.toFixed(2)),
        },
      ],
      series: [
        {},
        { label: 'val_loss', stroke: danger, width: 1.5, points: { show: true, size: 4 } },
        {
          label: 'val_acc',
          stroke: success,
          width: 1.5,
          scale: 'y2',
          points: { show: true, size: 4 },
        },
      ],
    };
  }

  function build() {
    if (!container) return;
    plot?.destroy();
    const xs = chartData.epochs.length === 0 ? [0] : chartData.epochs;
    const l = chartData.valLosses.length === 0 ? [null] : chartData.valLosses;
    const a = chartData.valAccs.length === 0 ? [null] : chartData.valAccs;
    plot = new uPlot(
      makeOpts(container.clientWidth, container.clientHeight),
      [xs, l, a] as AlignedData,
      container,
    );
  }

  $effect(() => {
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
    const xs = chartData.epochs.length === 0 ? [0] : chartData.epochs;
    const l = chartData.valLosses.length === 0 ? [null] : chartData.valLosses;
    const a = chartData.valAccs.length === 0 ? [null] : chartData.valAccs;
    plot.setData([xs, l, a] as AlignedData);
  });
</script>

<div class="card flex flex-col min-h-0 flex-1 min-w-0 overflow-hidden">
  <div class="px-3 pt-2 text-xs font-semibold text-[var(--color-muted)]">
    Validation (per epoch)
  </div>
  <div bind:this={container} class="flex-1 min-h-0"></div>
</div>

<script lang="ts">
  import { training } from '../../../state.svelte';

  let { classes }: { classes: string[] } = $props();

  // Probabilities for the currently-selected image (or null when nothing
  // selected / no session yet).
  let probs = $derived.by(() => {
    if (training.selectedIndex === null) return null;
    return training.predictions[training.selectedIndex] ?? null;
  });

  // Index of the highest-prob class (the model's prediction).
  let predictedIdx = $derived.by(() => {
    if (!probs) return null;
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    return best;
  });

  // Index of the actual label in the class list (for the highlight).
  let actualIdx = $derived.by(() => {
    if (training.selectedIndex === null) return null;
    const sample = training.batch[training.selectedIndex];
    if (!sample) return null;
    const idx = classes.indexOf(sample.label);
    return idx >= 0 ? idx : null;
  });

  // Track the chart container's pixel width so we can size bars to fit
  // exactly — no horizontal scroll, no preserveAspectRatio="none" text
  // distortion. svelte-bound clientWidth uses ResizeObserver under the
  // hood and updates reactively.
  let containerWidth = $state(0);

  let n = $derived(classes.length);
  const GAP = 1;
  const HEADER_H = 14;
  const FOOTER_H = 14;
  const PLOT_H = 120;
  const svgH = HEADER_H + PLOT_H + FOOTER_H;

  // SVG width tracks the container (1:1, so coordinates equal pixels).
  // Bar width is whatever's left after subtracting the inter-bar gaps —
  // can go subpixel for very large class counts; the browser handles AA.
  let svgW = $derived(Math.max(1, containerWidth));
  let BAR_W = $derived.by(() => {
    if (n <= 0 || svgW <= 0) return 0;
    return Math.max(0.5, (svgW - GAP) / n - GAP);
  });
</script>

<div class="flex-1 min-w-0 min-h-0 flex flex-col p-3 gap-2">
  <header class="flex items-baseline justify-between text-xs">
    <span class="font-semibold text-[var(--color-heading)]">Single training sample</span>
    {#if probs !== null && predictedIdx !== null && actualIdx !== null}
      <span class="text-[var(--color-muted)] font-mono">
        actual <span class="text-[var(--color-text)]">{classes[actualIdx]}</span>
        · predicted
        <span
          class="font-mono"
          style="color: {predictedIdx === actualIdx
            ? 'var(--color-success)'
            : 'var(--color-danger)'}"
        >{classes[predictedIdx]}</span>
        ({(probs[predictedIdx] * 100).toFixed(1)}%)
      </span>
    {:else if !probs}
      <span class="text-[var(--color-muted)]">
        — select an image
      </span>
    {/if}
  </header>

  <div bind:clientWidth={containerWidth} class="flex-1 min-h-0 min-w-0">
    <svg
      width={svgW}
      height={svgH}
      class="block"
      role="img"
      aria-label="Class probability distribution"
    >
      <!-- Plot baseline -->
      <line
        x1="0"
        y1={HEADER_H + PLOT_H}
        x2={svgW}
        y2={HEADER_H + PLOT_H}
        stroke="var(--color-border)"
        stroke-width="1"
      />

      {#each classes as label, i}
        {@const x = GAP + i * (BAR_W + GAP)}
        {@const p = probs ? probs[i] : 0}
        {@const h = Math.max(0, p * PLOT_H)}
        {@const y = HEADER_H + PLOT_H - h}
        {@const isPred = predictedIdx === i}
        {@const isActual = actualIdx === i}
        {@const fill = isActual && isPred
          ? 'var(--color-success)'
          : isActual
          ? 'var(--color-accent)'
          : isPred
          ? 'var(--color-danger)'
          : 'var(--color-muted)'}

        <rect
          x={x}
          y={y}
          width={BAR_W}
          height={h}
          fill={fill}
          opacity={p > 0 || isActual || isPred ? 1 : 0.2}
        >
          <title>
            {label}: {(p * 100).toFixed(2)}%
          </title>
        </rect>

        {#if isActual}
          <line
            x1={x - 0.5}
            y1={HEADER_H - 2}
            x2={x + BAR_W + 0.5}
            y2={HEADER_H - 2}
            stroke="var(--color-accent)"
            stroke-width="2"
          />
        {/if}
      {/each}

      <!-- A few tick labels along the x axis (every Nth class) -->
      {#each classes as _, i}
        {#if i % Math.max(1, Math.floor(n / 10)) === 0 || i === n - 1}
          {@const x = GAP + i * (BAR_W + GAP) + BAR_W / 2}
          <text
            x={x}
            y={svgH - 2}
            text-anchor="middle"
            font-size="8"
            fill="var(--color-muted)"
            style="font-family: var(--font-mono)"
          >{i + 1}</text>
        {/if}
      {/each}
    </svg>
  </div>
</div>

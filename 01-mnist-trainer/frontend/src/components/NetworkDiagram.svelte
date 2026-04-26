<script lang="ts">
  /**
   * Renders an architecture as a horizontal SVG schematic.
   *
   * Layers are positioned left→right at fixed x intervals. Each layer's
   * visual is dispatched on its `type` field — so adding a new layer type
   * (e.g., for user-designed architectures later) only requires:
   *   1. extending the `styleFor` map below for color/border, and
   *   2. (optional) adding a custom inner-glyph render in the {#if} chain.
   *
   * Connection rendering is currently a single straight arrow between
   * adjacent layers. A future enhancement could fan-out lines for the
   * textbook "fully-connected mesh" look between two `linear` layers —
   * the geometry stubs (leftPoint/rightPoint) below already isolate that.
   */
  import type { LayerSpec } from '../api';
  import { theme } from '../state.svelte';

  let { layers }: { layers: LayerSpec[] } = $props();

  // Force a re-render when the theme changes so SVG colors refresh.
  $effect(() => {
    theme.version;
  });

  const LAYER_W = 78;
  const LAYER_H = 84;
  const GAP = 26;
  const PAD = 12;

  type LayerStyle = { fill: string; stroke: string; accent: string };

  function styleFor(type: string): LayerStyle {
    switch (type) {
      case 'input':
        return {
          fill: 'var(--color-surface-2)',
          stroke: 'var(--color-muted)',
          accent: 'var(--color-muted)',
        };
      case 'output':
        return {
          fill: 'var(--color-accent)',
          stroke: 'var(--color-accent)',
          accent: 'var(--color-on-accent)',
        };
      case 'linear':
        return {
          fill: 'var(--color-surface-2)',
          stroke: 'var(--color-accent)',
          accent: 'var(--color-accent)',
        };
      case 'conv2d':
        return {
          fill: 'var(--color-surface-2)',
          stroke: 'var(--color-success)',
          accent: 'var(--color-success)',
        };
      case 'maxpool2d':
        return {
          fill: 'var(--color-surface)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-muted)',
        };
      case 'flatten':
        return {
          fill: 'var(--color-surface)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-muted)',
        };
      case 'activation':
        return {
          fill: 'transparent',
          stroke: 'var(--color-border)',
          accent: 'var(--color-danger)',
        };
      case 'dropout':
        return {
          fill: 'var(--color-surface)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-muted)',
        };
      default:
        return {
          fill: 'var(--color-surface-2)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-text)',
        };
    }
  }

  function shapeText(shape: number[]): string {
    return shape.join('×');
  }

  function sizeText(size: number): string {
    if (size >= 1_000_000) return `${(size / 1_000_000).toFixed(1)}M`;
    if (size >= 1_000) return `${(size / 1_000).toFixed(1)}k`;
    return String(size);
  }

  let totalW = $derived(layers.length * (LAYER_W + GAP) - GAP + PAD * 2);
  const totalH = LAYER_H + PAD * 2;

  // Centerpoint helpers for the connection arrows.
  function rightPoint(i: number) {
    return { x: PAD + i * (LAYER_W + GAP) + LAYER_W, y: PAD + LAYER_H / 2 };
  }
  function leftPoint(i: number) {
    return { x: PAD + i * (LAYER_W + GAP), y: PAD + LAYER_H / 2 };
  }
</script>

<div class="overflow-x-auto pb-1">
  <svg
    width={totalW}
    height={totalH}
    viewBox={`0 0 ${totalW} ${totalH}`}
    class="block"
    role="img"
    aria-label="Network architecture diagram"
  >
    <defs>
      <marker
        id="arr"
        markerWidth="6"
        markerHeight="6"
        refX="5"
        refY="3"
        orient="auto"
        markerUnits="strokeWidth"
      >
        <path d="M0 0 L6 3 L0 6 Z" fill="var(--color-muted)" />
      </marker>
    </defs>

    <!-- Connections (drawn first so they sit behind the layer rectangles) -->
    {#each layers.slice(0, -1) as _, i}
      {@const a = rightPoint(i)}
      {@const b = leftPoint(i + 1)}
      <line
        x1={a.x}
        y1={a.y}
        x2={b.x - 4}
        y2={b.y}
        stroke="var(--color-muted)"
        stroke-width="1"
        marker-end="url(#arr)"
        opacity="0.6"
      />
    {/each}

    <!-- Layers -->
    {#each layers as l, i (i)}
      {@const x = PAD + i * (LAYER_W + GAP)}
      {@const y = PAD}
      {@const s = styleFor(l.type)}

      <g transform={`translate(${x}, ${y})`}>
        <rect
          width={LAYER_W}
          height={LAYER_H}
          rx="6"
          fill={s.fill}
          stroke={s.stroke}
          stroke-width="1.25"
        />

        <!-- Type label -->
        <text
          x={LAYER_W / 2}
          y="16"
          text-anchor="middle"
          font-size="10"
          font-weight="600"
          fill={s.accent}
          style="font-family: var(--font-sans); letter-spacing: 0.02em"
        >
          {l.label}
        </text>

        <!-- Inner glyph: subtle visual differentiation per type -->
        {#if l.type === 'linear' || l.type === 'input' || l.type === 'output'}
          <!-- Column of dots representing neurons -->
          {@const n = Math.min(5, l.size)}
          {#each Array(n) as _, di}
            <circle
              cx={LAYER_W / 2}
              cy={32 + di * 7}
              r="2"
              fill={s.accent}
              opacity={l.type === 'output' ? 1 : 0.55}
            />
          {/each}
        {:else if l.type === 'conv2d' || l.type === 'maxpool2d'}
          <!-- Stacked rectangles suggesting feature-map depth -->
          {@const baseY = 30}
          {@const w = 30}
          {@const h = 22}
          <rect x={LAYER_W / 2 - w / 2 - 4} y={baseY - 4} width={w} height={h} rx="2"
            fill="none" stroke={s.accent} stroke-width="1" opacity="0.4" />
          <rect x={LAYER_W / 2 - w / 2 - 2} y={baseY - 2} width={w} height={h} rx="2"
            fill="none" stroke={s.accent} stroke-width="1" opacity="0.65" />
          <rect x={LAYER_W / 2 - w / 2} y={baseY} width={w} height={h} rx="2"
            fill="var(--color-bg)" stroke={s.accent} stroke-width="1.25" />
        {:else if l.type === 'flatten'}
          <!-- Funnel/transition shape -->
          <path
            d={`M ${LAYER_W / 2 - 14} 32 L ${LAYER_W / 2 + 14} 38 L ${LAYER_W / 2 + 14} 50 L ${LAYER_W / 2 - 14} 56 Z`}
            fill="none"
            stroke={s.accent}
            stroke-width="1"
            opacity="0.7"
          />
        {:else if l.type === 'activation'}
          <!-- ReLU-ish hockey-stick path -->
          <path
            d={`M ${LAYER_W / 2 - 14} 50 L ${LAYER_W / 2} 50 L ${LAYER_W / 2 + 14} 32`}
            fill="none"
            stroke={s.accent}
            stroke-width="1.5"
          />
        {:else if l.type === 'dropout'}
          <!-- Sparse dots -->
          {#each [0, 1, 2, 3, 4] as di}
            {@const cx = LAYER_W / 2 + (di - 2) * 7}
            <circle cx={cx} cy="44" r="2" fill={s.accent} opacity={di === 1 || di === 3 ? 0.15 : 0.6} />
          {/each}
        {/if}

        <!-- Shape & size labels at the bottom -->
        <text
          x={LAYER_W / 2}
          y={LAYER_H - 22}
          text-anchor="middle"
          font-size="11"
          font-weight="600"
          fill="var(--color-text)"
          style="font-family: var(--font-mono)"
        >
          {shapeText(l.shape)}
        </text>
        <text
          x={LAYER_W / 2}
          y={LAYER_H - 8}
          text-anchor="middle"
          font-size="9"
          fill="var(--color-muted)"
          style="font-family: var(--font-mono)"
        >
          {sizeText(l.size)} elem
        </text>
      </g>
    {/each}
  </svg>
</div>

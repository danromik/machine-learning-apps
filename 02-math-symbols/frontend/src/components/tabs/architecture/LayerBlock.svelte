<script lang="ts">
  import type { LayerType } from './computeArchitecture';
  import { theme } from '../../../state.svelte';

  // 'input' / 'output' for the boundary blocks, otherwise the user-layer type.
  export type BlockType = 'input' | 'output' | LayerType;

  let {
    type,
    label,
    primary,
    shape,
    error,
    onDelete,
  }: {
    type: BlockType;
    label: string;
    primary?: string;
    shape?: number[] | null;
    error?: string;
    onDelete?: () => void;
  } = $props();

  // Force a re-paint of the SVG glyph when the theme changes — the strokes
  // use CSS custom properties whose computed values shift between themes.
  $effect(() => {
    theme.version;
  });

  type Style = { fill: string; stroke: string; accent: string };

  function styleFor(t: BlockType): Style {
    switch (t) {
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
      case 'relu':
        return {
          fill: 'var(--color-surface)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-danger)',
        };
      case 'dropout':
        return {
          fill: 'var(--color-surface)',
          stroke: 'var(--color-border)',
          accent: 'var(--color-muted)',
        };
    }
  }

  let style = $derived(styleFor(type));
  // 'middle' iff this is a user-droppable layer (everything other than the
  // fixed Input/Output boundary blocks). The drop-index calculation uses
  // [data-block-kind="middle"] to find user layers in the DOM.
  let blockKind = $derived(
    type === 'input' ? 'input' : type === 'output' ? 'output' : 'middle'
  );
</script>

<div
  class="relative shrink-0 rounded-lg border-2 flex flex-col items-center"
  style="
    width: 96px;
    height: 108px;
    border-color: {style.stroke};
    background: {style.fill};
    padding: 6px 8px 6px;
  "
  data-block-kind={blockKind}
>
  {#if onDelete}
    <button
      type="button"
      class="absolute top-0.5 right-0.5 w-4 h-4 flex items-center justify-center rounded
             text-[14px] leading-none text-[var(--color-muted)]
             hover:text-[var(--color-danger)] hover:bg-[var(--color-surface-2)]"
      onclick={onDelete}
      title="Delete layer"
      aria-label="Delete layer"
    >×</button>
  {/if}

  <!-- Type label -->
  <div
    class="text-[10px] font-semibold uppercase tracking-wide leading-none mb-1"
    style="color: {style.accent}"
  >
    {label}
  </div>

  <!-- Inner glyph (38×30 SVG centered) -->
  <svg width="44" height="32" viewBox="0 0 44 32" class="block">
    {#if type === 'conv2d' || type === 'maxpool2d'}
      <!-- stacked feature-map rectangles -->
      <rect x="6" y="2" width="28" height="20" rx="2"
        fill="none" stroke={style.accent} stroke-width="1" opacity="0.4" />
      <rect x="9" y="5" width="28" height="20" rx="2"
        fill="none" stroke={style.accent} stroke-width="1" opacity="0.65" />
      <rect x="12" y="8" width="28" height="20" rx="2"
        fill="var(--color-bg)" stroke={style.accent} stroke-width="1.25" />
    {:else if type === 'linear' || type === 'input' || type === 'output'}
      <!-- column of dots = neurons -->
      {#each [0, 1, 2, 3, 4] as di}
        <circle cx="22" cy={4 + di * 6} r="2"
          fill={style.accent} opacity={type === 'output' ? 1 : 0.55} />
      {/each}
    {:else if type === 'flatten'}
      <!-- funnel-shaped transition -->
      <path d="M 6 4 L 38 12 L 38 20 L 6 28 Z"
        fill="none" stroke={style.accent} stroke-width="1" opacity="0.7" />
    {:else if type === 'relu'}
      <!-- hockey-stick rectifier -->
      <path d="M 6 26 L 22 26 L 38 4"
        fill="none" stroke={style.accent} stroke-width="1.5" />
    {:else if type === 'dropout'}
      <!-- sparse dots, some dimmed -->
      {#each [0, 1, 2, 3, 4] as di}
        <circle
          cx={6 + di * 8} cy="16" r="2"
          fill={style.accent}
          opacity={di === 1 || di === 3 ? 0.15 : 0.6}
        />
      {/each}
    {/if}
  </svg>

  <!-- Primary value (e.g. "32") -->
  {#if primary !== undefined}
    <div
      class="text-sm font-mono leading-none mt-0.5"
      style="color: {style.accent}"
    >
      {primary}
    </div>
  {:else}
    <div class="text-sm font-mono leading-none mt-0.5 text-transparent select-none">·</div>
  {/if}

  <!-- Shape / error -->
  <div class="text-[10px] text-[var(--color-muted)] font-mono mt-auto leading-none">
    {#if error}
      <span class="text-[var(--color-danger)]" title={error}>! shape</span>
    {:else if shape}
      {shape.join('×')}
    {:else}
      &nbsp;
    {/if}
  </div>
</div>

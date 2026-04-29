<script lang="ts">
  import { architecture, applyArchitecturePreset, clearPreset } from '../../../state.svelte';
  import type { ArchitecturePreset } from '../../../api';

  function onApply(p: ArchitecturePreset) {
    applyArchitecturePreset(p);
  }

  function onCustom() {
    clearPreset();
    architecture.layers = [];
  }
</script>

<div
  class="border-b border-[var(--color-border)] px-3 py-2 flex items-center gap-2 flex-wrap
         bg-[var(--color-surface)]/40"
>
  <span class="text-xs font-semibold text-[var(--color-muted)] uppercase tracking-wide pr-1">
    Preset
  </span>
  {#each architecture.presets as p (p.name)}
    {@const active = architecture.preset === p.name}
    <button
      type="button"
      class="preset-button"
      class:preset-button-active={active}
      onclick={() => onApply(p)}
      title={p.description}
    >
      <span class="preset-label">{p.label}</span>
      <span class="preset-year">{p.year}</span>
      {#if p.locked}
        <span class="preset-lock" title="Locked architecture (no editing)">🔒</span>
      {/if}
    </button>
  {/each}
  <button
    type="button"
    class="preset-button"
    class:preset-button-active={architecture.preset === null}
    onclick={onCustom}
    title="Build from scratch with the drag-and-drop canvas"
  >
    <span class="preset-label">Custom</span>
    <span class="preset-year">your design</span>
  </button>

  {#if architecture.presets.find((p) => p.name === architecture.preset)}
    {@const cur = architecture.presets.find((p) => p.name === architecture.preset)!}
    <p class="text-xs text-[var(--color-muted)] leading-snug ml-3 flex-1 min-w-0 max-w-2xl">
      {cur.tagline}
    </p>
  {/if}
</div>

<style>
  .preset-button {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.625rem;
    font-size: 0.75rem;
    border: 1px solid var(--color-border);
    border-radius: 0.375rem;
    background: var(--color-surface);
    color: var(--color-text);
    cursor: pointer;
    transition: background 80ms, border-color 80ms;
  }
  .preset-button:hover {
    background: var(--color-surface-2);
  }
  .preset-button-active {
    border-color: var(--color-accent);
    background: var(--color-accent);
    color: var(--color-on-accent);
  }
  .preset-button-active:hover {
    background: var(--color-accent);
  }
  .preset-label {
    font-weight: 600;
  }
  .preset-year {
    font-size: 0.625rem;
    opacity: 0.7;
    font-family: var(--font-mono);
  }
  .preset-lock {
    font-size: 0.625rem;
  }
</style>

<script lang="ts">
  import {
    architecture,
    synthesis,
    INPUT_SHAPE,
    dragState,
  } from '../../../state.svelte';
  import {
    LAYER_TEMPLATES,
    suggestArchitecture,
    formatCount,
    type Layer,
    type LayerType,
    type ArchitectureComputed,
  } from './computeArchitecture';
  import LayerBlock from './LayerBlock.svelte';
  import Icon from '../../Icon.svelte';

  let { computed }: { computed: ArchitectureComputed } = $props();

  // ── refs + measurements ─────────────────────────────────────────────────

  let containerEl: HTMLDivElement | undefined = $state();
  let chainEl: HTMLDivElement | undefined = $state();
  let containerW = $state(0);
  let chainW = $state(0);
  let chainH = $state(0);

  // ── zoom state ──────────────────────────────────────────────────────────

  let zoom = $state(1);
  // True until the user manually adjusts zoom — keeps the diagram auto-fitted
  // as layers are added/removed. Cleared on +/− clicks; restored by Fit.
  let autoFit = $state(true);
  const ZOOM_MIN = 0.2;
  const ZOOM_MAX = 3;
  const ZOOM_STEP = 1.25;
  // Reserve some left/right room inside the container so the chain doesn't
  // touch the padding edges when fitted.
  const FIT_PADDING = 16;

  $effect(() => {
    if (!containerEl) return;
    const ro = new ResizeObserver((entries) => {
      containerW = entries[0].contentRect.width;
    });
    ro.observe(containerEl);
    return () => ro.disconnect();
  });

  $effect(() => {
    if (!chainEl) return;
    const ro = new ResizeObserver((entries) => {
      chainW = entries[0].contentRect.width;
      chainH = entries[0].contentRect.height;
    });
    ro.observe(chainEl);
    return () => ro.disconnect();
  });

  // Recompute auto-fit zoom whenever the relevant inputs change.
  $effect(() => {
    if (!autoFit || containerW <= 0 || chainW <= 0) return;
    const avail = Math.max(0, containerW - FIT_PADDING * 2);
    if (chainW > avail) zoom = avail / chainW;
    else zoom = 1;
  });

  function zoomIn() {
    autoFit = false;
    zoom = Math.min(ZOOM_MAX, zoom * ZOOM_STEP);
  }
  function zoomOut() {
    autoFit = false;
    zoom = Math.max(ZOOM_MIN, zoom / ZOOM_STEP);
  }
  function fit() {
    autoFit = true;
    // The effect will recompute zoom on its next run.
  }

  // ── drag/drop ───────────────────────────────────────────────────────────

  let dropIndex = $state<number | null>(null);

  $effect(() => {
    if (dragState.draggingType === null) {
      dropIndex = null;
    }
  });

  // Count how many user-layer block centers are to the left of cursor.x.
  // getBoundingClientRect returns visual coordinates so this works correctly
  // even with the zoom transform applied.
  function computeDropIndex(clientX: number): number {
    if (!chainEl) return architecture.layers.length;
    const blocks = chainEl.querySelectorAll<HTMLElement>(
      '[data-block-kind="middle"]'
    );
    let idx = 0;
    for (const el of blocks) {
      const rect = el.getBoundingClientRect();
      if (clientX > rect.left + rect.width / 2) idx++;
      else break;
    }
    return Math.min(idx, blocks.length);
  }

  function onDragOver(e: DragEvent) {
    if (!e.dataTransfer?.types.includes('application/x-layer-type')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    dropIndex = computeDropIndex(e.clientX);
  }

  function onDragLeave(e: DragEvent) {
    const related = e.relatedTarget as Node | null;
    if (!containerEl || (related && containerEl.contains(related))) return;
    dropIndex = null;
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    const type = e.dataTransfer?.getData('application/x-layer-type') as LayerType;
    if (LAYER_TEMPLATES[type] && dropIndex !== null) {
      const layer: Layer = {
        id: newId(),
        type,
        params: { ...LAYER_TEMPLATES[type].defaults },
      };
      architecture.layers.splice(dropIndex, 0, layer);
    }
    dropIndex = null;
  }

  // ── layer mutations ─────────────────────────────────────────────────────

  function newId(): string {
    return crypto.randomUUID
      ? crypto.randomUUID()
      : `l-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function deleteLayer(id: string) {
    architecture.layers = architecture.layers.filter((l) => l.id !== id);
  }

  function clearLayers() {
    architecture.layers = [];
    architecture.suggestionReasoning = null;
  }

  // ── numClasses + numTrainFonts derived from synthesis ──────────────────

  let numClasses = $derived.by(() => {
    if (!synthesis.loaded || synthesis.categories.length === 0) return 10;
    let n = 0;
    for (const c of synthesis.categories) {
      if (synthesis.selectedCategories[c.id]) n += c.count;
    }
    return n || 10;
  });

  let numTrainFonts = $derived.by(() => {
    if (!synthesis.loaded) return 1;
    const n = synthesis.fonts.filter(
      (f) => synthesis.fontUsage[f.family] === 'train'
    ).length;
    return Math.max(1, n);
  });

  function applySuggestion() {
    const { layers, reasoning } = suggestArchitecture({
      numClasses,
      numTrainFonts,
      noiseEnabled: synthesis.augmentation.noise.enabled,
      noiseLevel: synthesis.augmentation.noise.max_level,
      skewEnabled: synthesis.augmentation.skew.enabled,
    });
    architecture.layers = layers;
    architecture.suggestionReasoning = reasoning;
  }

  // ── layer-block rendering helpers ──────────────────────────────────────

  function primaryFor(
    type: LayerType,
    params: Record<string, number>
  ): string | undefined {
    const tmpl = LAYER_TEMPLATES[type];
    if (!tmpl.primary) return undefined;
    const v = params[tmpl.primary.key];
    if (v === undefined) return undefined;
    return tmpl.primary.format ? tmpl.primary.format(v) : String(v);
  }

  // Visual size of the scaled chain (used to size the spacer so overflow-auto
  // computes correctly — transform: scale doesn't change the layout box).
  let scaledW = $derived(chainW * zoom);
  let scaledH = $derived(chainH * zoom);
</script>

<div class="relative flex-1 min-h-0 min-w-0 flex flex-col">
  <!-- Scrollable diagram area. Outer container has the static padding so the
       padding stays at the edges while the inner chain pans/scrolls. -->
  <div
    class="flex-1 min-h-0 overflow-auto px-12 pt-20 pb-24 flex items-start"
    style="justify-content: safe center; min-height: 0"
    bind:this={containerEl}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
    ondrop={onDrop}
    role="region"
    aria-label="Architecture diagram"
  >
    <!-- Spacer takes the *visual* dimensions of the scaled chain so the
         outer overflow-auto sees the right size. -->
    <div
      class="shrink-0 relative"
      style="width: {Math.max(scaledW, 1)}px; height: {Math.max(scaledH, 1)}px;"
    >
      <div
        bind:this={chainEl}
        class="absolute top-0 left-0 flex items-center gap-2"
        style="transform: scale({zoom}); transform-origin: top left;"
      >
        <LayerBlock type="input" label="Input" shape={INPUT_SHAPE} />

        {#each architecture.layers as layer, i (layer.id)}
          {@render arrowSlot(i)}
          <LayerBlock
            type={layer.type}
            label={LAYER_TEMPLATES[layer.type].label}
            primary={primaryFor(layer.type, layer.params)}
            shape={computed.shapes[i + 1]}
            error={computed.errors[i]}
            onDelete={() => deleteLayer(layer.id)}
          />
        {/each}

        {@render arrowSlot(architecture.layers.length)}
        <LayerBlock
          type="output"
          label="Output"
          primary={String(numClasses)}
          shape={[numClasses]}
          error={computed.outputValid
            ? undefined
            : 'final layer must be 1D — add Flatten + Linear before Output'}
        />
      </div>
    </div>
  </div>

  {#if architecture.layers.length === 0 && dropIndex === null}
    <div
      class="absolute inset-x-0 top-1/2 -translate-y-2 text-center
             text-sm text-[var(--color-muted)] pointer-events-none"
    >
      Drag layers from the sidebar — or click Suggest Neural Network.
    </div>
  {/if}

  <!-- Top-left: architecture stats (layers + total weights) -->
  <div
    class="absolute top-2 left-2 inline-flex items-center gap-3
           text-xs text-[var(--color-muted)]
           px-2.5 py-1 rounded
           border border-[var(--color-border)]
           bg-[var(--color-surface)]/90 shadow-sm"
  >
    <span>
      Layers:
      <span class="font-mono text-[var(--color-text)] tabular-nums"
        >{architecture.layers.length}</span
      >
    </span>
    <span>
      Weights:
      <span class="font-mono text-[var(--color-text)] tabular-nums"
        >{formatCount(computed.totalParams)}</span
      >
    </span>
  </div>

  <!-- Top-right: zoom controls -->
  <div
    class="absolute top-2 right-2 inline-flex rounded
           border border-[var(--color-border)] overflow-hidden
           bg-[var(--color-surface)]/90 shadow-sm"
  >
    <button
      type="button"
      class="px-2 py-1 text-sm font-mono leading-none
             text-[var(--color-text)]
             hover:bg-[var(--color-surface-2)]"
      onclick={zoomOut}
      title="Zoom out"
      aria-label="Zoom out"
    >−</button>
    <button
      type="button"
      class="px-2 py-1 text-sm font-mono leading-none
             text-[var(--color-text)]
             border-l border-[var(--color-border)]
             hover:bg-[var(--color-surface-2)]"
      onclick={zoomIn}
      title="Zoom in"
      aria-label="Zoom in"
    >+</button>
    <button
      type="button"
      class="px-2 py-1 text-xs font-mono leading-none
             text-[var(--color-text)]
             border-l border-[var(--color-border)]
             {autoFit ? 'bg-[var(--color-surface-2)]' : ''}
             hover:bg-[var(--color-surface-2)]"
      onclick={fit}
      title="Fit to view"
      aria-label="Fit to view"
    >Fit</button>
    <span
      class="px-2 py-1 text-[10px] font-mono leading-none
             text-[var(--color-muted)]
             border-l border-[var(--color-border)]
             flex items-center"
      aria-label="Current zoom level"
    >{Math.round(zoom * 100)}%</span>
  </div>

  <!-- Bottom row: reasoning on the left, Suggest + Clear on the right.
       The button group has ml-auto so it stays anchored to the right edge
       even when the reasoning hits its max-width and leaves leftover space
       (otherwise default flex-start packing would pull the buttons left). -->
  <div
    class="absolute bottom-3 left-3 right-3 flex items-end gap-3
           pointer-events-none"
  >
    {#if architecture.suggestionReasoning}
      <div
        class="pointer-events-auto text-xs text-[var(--color-muted)] leading-snug
               flex-1 min-w-0 max-w-3xl
               px-2.5 py-1.5 rounded
               bg-[var(--color-surface)]/90 border border-[var(--color-border)]"
      >
        {architecture.suggestionReasoning}
      </div>
    {/if}
    <div class="flex gap-2 pointer-events-auto shrink-0 ml-auto">
      <button
        type="button"
        class="px-2.5 py-1 text-xs font-medium rounded btn-outline"
        onclick={applySuggestion}
        title="Generate a CNN architecture matched to the current Data Synthesis settings"
      >
        Suggest Neural Network
      </button>
      <button
        type="button"
        class="px-2.5 py-1 text-xs font-medium rounded btn-outline"
        onclick={clearLayers}
        disabled={architecture.layers.length === 0}
        title="Remove all layers from the diagram"
      >
        Clear
      </button>
    </div>
  </div>
</div>

{#snippet arrowSlot(index: number)}
  {@const active = dropIndex === index && dragState.draggingType !== null}
  <div
    class="relative flex items-center justify-center shrink-0 select-none w-8 h-12"
  >
    <span
      class="text-xl text-[var(--color-muted)] transition-opacity"
      class:opacity-0={active}>→</span
    >
    {#if active && dragState.draggingType}
      <div
        class="absolute z-10 px-2 py-1 rounded
               bg-[var(--color-accent)] text-[var(--color-on-accent)]
               text-[11px] font-mono font-semibold whitespace-nowrap
               pointer-events-none shadow"
      >
        + {LAYER_TEMPLATES[dragState.draggingType].label}
      </div>
    {/if}
  </div>
{/snippet}

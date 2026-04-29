<script lang="ts">
  import {
    architecture,
    dataset,
    INPUT_SHAPE,
    dragState,
    clearPreset,
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

  let {
    computed,
    numClasses,
  }: { computed: ArchitectureComputed; numClasses: number } = $props();

  // ── refs + measurements ─────────────────────────────────────────────────

  let containerEl: HTMLDivElement | undefined = $state();
  let chainEl: HTMLDivElement | undefined = $state();
  let containerW = $state(0);
  let chainW = $state(0);
  let chainH = $state(0);

  // ── zoom state ──────────────────────────────────────────────────────────

  let zoom = $state(1);
  let autoFit = $state(true);
  const ZOOM_MIN = 0.2;
  const ZOOM_MAX = 3;
  const ZOOM_STEP = 1.25;
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
  }

  // ── drag/drop ───────────────────────────────────────────────────────────

  let dropIndex = $state<number | null>(null);

  $effect(() => {
    if (dragState.draggingType === null) {
      dropIndex = null;
    }
  });

  let locked = $derived(architecture.preset === 'resnet18');

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
    if (locked) return;
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
    if (locked) return;
    e.preventDefault();
    const type = e.dataTransfer?.getData('application/x-layer-type') as LayerType;
    if (LAYER_TEMPLATES[type] && dropIndex !== null) {
      const layer: Layer = {
        id: newId(),
        type,
        params: { ...LAYER_TEMPLATES[type].defaults },
      };
      architecture.layers.splice(dropIndex, 0, layer);
      // Once the user has edited a preset, it's no longer "the preset" —
      // clear the name so the preset selector doesn't claim ownership.
      if (architecture.preset !== null) clearPreset();
    }
    dropIndex = null;
  }

  function newId(): string {
    return crypto.randomUUID
      ? crypto.randomUUID()
      : `l-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function deleteLayer(id: string) {
    architecture.layers = architecture.layers.filter((l) => l.id !== id);
    if (architecture.preset !== null) clearPreset();
  }

  function clearLayers() {
    architecture.layers = [];
    clearPreset();
  }

  function applySuggestion() {
    const { layers } = suggestArchitecture({ numClasses });
    architecture.layers = layers;
    clearPreset();
  }

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

  let scaledW = $derived(chainW * zoom);
  let scaledH = $derived(chainH * zoom);
</script>

<div class="relative flex-1 min-h-0 min-w-0 flex flex-col">
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

        {#if locked}
          {@render arrowSlot(0)}
          <!-- Locked preset placeholder block — single visual stand-in for
               the residual graph that doesn't fit the linear drag-and-drop
               representation. -->
          <div
            class="locked-block"
            data-block-kind="locked"
            title="ResNet-18 (torchvision) — locked architecture; switch to Custom to edit"
          >
            <div class="locked-label">ResNet-18</div>
            <div class="locked-meta">torchvision · residual</div>
            <div class="locked-meta">11.2M params</div>
          </div>
          {@render arrowSlot(0)}
        {:else}
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
        {/if}

        <LayerBlock
          type="output"
          label="Output"
          primary={String(numClasses)}
          shape={[numClasses]}
          error={!locked && !computed.outputValid
            ? 'final layer must be 1D — add Flatten + Linear before Output'
            : undefined}
        />
      </div>
    </div>
  </div>

  {#if architecture.layers.length === 0 && !locked && dropIndex === null}
    <div
      class="absolute inset-x-0 top-1/2 -translate-y-2 text-center
             text-sm text-[var(--color-muted)] pointer-events-none"
    >
      Drag layers from the sidebar — or pick a preset above.
    </div>
  {/if}

  <!-- Top-left: stats -->
  <div
    class="absolute top-2 left-2 inline-flex items-center gap-3
           text-xs text-[var(--color-muted)]
           px-2.5 py-1 rounded
           border border-[var(--color-border)]
           bg-[var(--color-surface)]/90 shadow-sm"
  >
    {#if locked}
      <span>Layers:
        <span class="font-mono text-[var(--color-text)] tabular-nums">18</span>
      </span>
      <span>Weights:
        <span class="font-mono text-[var(--color-text)] tabular-nums">11.2M</span>
      </span>
    {:else}
      <span>Layers:
        <span class="font-mono text-[var(--color-text)] tabular-nums"
          >{architecture.layers.length}</span
        >
      </span>
      <span>Weights:
        <span class="font-mono text-[var(--color-text)] tabular-nums"
          >{formatCount(computed.totalParams)}</span
        >
      </span>
    {/if}
  </div>

  <!-- Top-right: zoom controls -->
  <div
    class="absolute top-2 right-2 inline-flex rounded
           border border-[var(--color-border)] overflow-hidden
           bg-[var(--color-surface)]/90 shadow-sm"
  >
    <button type="button" class="px-2 py-1 text-sm font-mono leading-none text-[var(--color-text)] hover:bg-[var(--color-surface-2)]"
      onclick={zoomOut} title="Zoom out" aria-label="Zoom out">−</button>
    <button type="button" class="px-2 py-1 text-sm font-mono leading-none text-[var(--color-text)] border-l border-[var(--color-border)] hover:bg-[var(--color-surface-2)]"
      onclick={zoomIn} title="Zoom in" aria-label="Zoom in">+</button>
    <button type="button" class="px-2 py-1 text-xs font-mono leading-none text-[var(--color-text)] border-l border-[var(--color-border)] {autoFit ? 'bg-[var(--color-surface-2)]' : ''} hover:bg-[var(--color-surface-2)]"
      onclick={fit} title="Fit to view" aria-label="Fit to view">Fit</button>
    <span class="px-2 py-1 text-[10px] font-mono leading-none text-[var(--color-muted)] border-l border-[var(--color-border)] flex items-center"
      aria-label="Current zoom level">{Math.round(zoom * 100)}%</span>
  </div>

  <!-- Bottom row: Suggest + Clear (custom mode only) -->
  {#if !locked}
    <div
      class="absolute bottom-3 left-3 right-3 flex items-end gap-3 pointer-events-none"
    >
      <div class="flex gap-2 pointer-events-auto shrink-0 ml-auto">
        <button
          type="button"
          class="px-2.5 py-1 text-xs font-medium rounded btn-outline"
          onclick={applySuggestion}
          title="Generate a generic CNN matched to the input shape and class count"
        >
          Suggest CNN
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
  {/if}
</div>

{#snippet arrowSlot(index: number)}
  {@const active = !locked && dropIndex === index && dragState.draggingType !== null}
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

<style>
  .locked-block {
    width: 9rem;
    height: 6.75rem;
    border: 2px dashed var(--color-accent);
    border-radius: 0.5rem;
    background: var(--color-surface-2);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.25rem;
  }
  .locked-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--color-accent);
    font-family: var(--font-mono);
  }
  .locked-meta {
    font-size: 0.6rem;
    color: var(--color-muted);
    font-family: var(--font-mono);
  }
</style>

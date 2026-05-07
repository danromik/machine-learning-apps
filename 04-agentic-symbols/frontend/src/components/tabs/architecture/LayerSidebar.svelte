<script lang="ts">
  import {
    LAYER_ORDER,
    LAYER_TEMPLATES,
    type LayerType,
  } from './computeArchitecture';
  import { dragState } from '../../../state.svelte';

  function onDragStart(e: DragEvent, type: LayerType) {
    if (!e.dataTransfer) return;
    e.dataTransfer.setData('application/x-layer-type', type);
    e.dataTransfer.effectAllowed = 'copy';
    dragState.draggingType = type;
  }

  function onDragEnd() {
    dragState.draggingType = null;
  }
</script>

<aside
  class="w-32 shrink-0 border-r border-[var(--color-border)] p-2 flex flex-col gap-2 overflow-auto"
>
  <h4
    class="text-[10px] font-semibold text-[var(--color-muted)] uppercase tracking-wide px-1"
  >
    Drag to add
  </h4>
  <div class="flex flex-col gap-1.5">
    {#each LAYER_ORDER as type}
      {@const tmpl = LAYER_TEMPLATES[type]}
      <div
        class="px-2 py-1.5 text-sm rounded border border-[var(--color-border)]
               bg-[var(--color-surface)] hover:bg-[var(--color-surface-2)]
               cursor-grab active:cursor-grabbing select-none"
        draggable="true"
        ondragstart={(e) => onDragStart(e, type)}
        ondragend={onDragEnd}
        title={tmpl.description}
        role="button"
        tabindex="0"
        aria-label="Add {tmpl.label} layer"
      >
        {tmpl.label}
      </div>
    {/each}
  </div>
</aside>

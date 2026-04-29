<script lang="ts">
  import {
    LAYER_ORDER,
    LAYER_TEMPLATES,
    type LayerType,
  } from './computeArchitecture';
  import { dragState } from '../../../state.svelte';

  let { disabled = false }: { disabled?: boolean } = $props();

  function onDragStart(e: DragEvent, type: LayerType) {
    if (disabled) {
      e.preventDefault();
      return;
    }
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
  class:opacity-50={disabled}
>
  <h4
    class="text-[10px] font-semibold text-[var(--color-muted)] uppercase tracking-wide px-1"
  >
    {disabled ? 'Locked' : 'Drag to add'}
  </h4>
  <div class="flex flex-col gap-1.5">
    {#each LAYER_ORDER as type}
      {@const tmpl = LAYER_TEMPLATES[type]}
      <div
        class="px-2 py-1.5 text-sm rounded border border-[var(--color-border)]
               bg-[var(--color-surface)] hover:bg-[var(--color-surface-2)]
               select-none"
        class:cursor-grab={!disabled}
        class:active:cursor-grabbing={!disabled}
        class:cursor-not-allowed={disabled}
        draggable={disabled ? 'false' : 'true'}
        ondragstart={(e) => onDragStart(e, type)}
        ondragend={onDragEnd}
        title={disabled ? 'Architecture is locked — switch to Custom to edit' : tmpl.description}
        role="button"
        tabindex="0"
        aria-label="Add {tmpl.label} layer"
      >
        {tmpl.label}
      </div>
    {/each}
  </div>
</aside>

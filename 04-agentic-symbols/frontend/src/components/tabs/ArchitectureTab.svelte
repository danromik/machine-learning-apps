<script lang="ts">
  import { architecture, synthesis, INPUT_SHAPE } from '../../state.svelte';
  import { computeArchitecture } from './architecture/computeArchitecture';
  import LayerSidebar from './architecture/LayerSidebar.svelte';
  import ArchitectureDiagram from './architecture/ArchitectureDiagram.svelte';

  // numClasses comes from data synthesis. If synthesis hasn't been touched
  // yet (or no categories selected), default to 10 so the diagram shows
  // something sensible rather than 0.
  let numClasses = $derived.by(() => {
    if (!synthesis.loaded || synthesis.categories.length === 0) return 10;
    let n = 0;
    for (const c of synthesis.categories) {
      if (synthesis.selectedCategories[c.id]) n += c.count;
    }
    return n || 10;
  });

  let computed = $derived(
    computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
  );
</script>

<div class="h-full flex">
  <LayerSidebar />
  <ArchitectureDiagram {computed} />
</div>

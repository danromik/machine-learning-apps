<script lang="ts">
  import { architecture, dataset, INPUT_SHAPE } from '../../state.svelte';
  import { computeArchitecture } from './architecture/computeArchitecture';
  import LayerSidebar from './architecture/LayerSidebar.svelte';
  import ArchitectureDiagram from './architecture/ArchitectureDiagram.svelte';
  import PresetSelector from './architecture/PresetSelector.svelte';

  let numClasses = $derived(dataset.classes.length || 10);

  let computed = $derived(
    computeArchitecture(architecture.layers, INPUT_SHAPE, numClasses)
  );
</script>

<div class="h-full flex flex-col">
  <PresetSelector />
  <div class="flex-1 min-h-0 flex">
    <LayerSidebar disabled={architecture.preset === 'resnet18'} />
    <ArchitectureDiagram {computed} {numClasses} />
  </div>
</div>

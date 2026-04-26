<script lang="ts">
  import { synthesis, applyPreset } from '../../state.svelte';
  import { api } from '../../api';
  import PresetSelector from './data/PresetSelector.svelte';
  import SymbolsSection from './data/SymbolsSection.svelte';
  import FontsSection from './data/FontsSection.svelte';
  import AugmentationSection from './data/AugmentationSection.svelte';
  import TestSynthesisModal from './data/TestSynthesisModal.svelte';

  let loadError = $state<string | null>(null);
  let testModalOpen = $state(false);

  $effect(() => {
    if (synthesis.loaded) return;
    (async () => {
      try {
        const [{ categories }, { fonts }, intermediate] = await Promise.all([
          api.symbols(),
          api.fonts(),
          api.preset('intermediate'),
        ]);
        synthesis.categories = categories;
        synthesis.fonts = fonts;
        applyPreset(intermediate);
        synthesis.loaded = true;
      } catch (e) {
        loadError = (e as Error).message || String(e);
      }
    })();
  });
</script>

<div class="h-full overflow-auto">
  <div class="max-w-3xl mx-auto px-4 py-3 flex flex-col gap-2">
    {#if loadError}
      <div class="card p-3 border-[var(--color-danger)] text-[var(--color-danger)] text-sm">
        Failed to load synthesis config: {loadError}
      </div>
    {:else if !synthesis.loaded}
      <div class="card p-6 text-center text-sm text-[var(--color-muted)]">
        Loading symbols and fonts…
      </div>
    {:else}
      <PresetSelector onTestSynthesis={() => (testModalOpen = true)} />
      <SymbolsSection />
      <FontsSection />
      <AugmentationSection />
    {/if}
  </div>
</div>

{#if testModalOpen}
  <TestSynthesisModal onClose={() => (testModalOpen = false)} />
{/if}

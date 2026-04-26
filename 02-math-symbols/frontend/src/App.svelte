<script lang="ts">
  import Header from './components/Header.svelte';
  import StatusBar from './components/StatusBar.svelte';
  import OrientationTab from './components/tabs/OrientationTab.svelte';
  import DataSynthesisTab from './components/tabs/DataSynthesisTab.svelte';
  import ArchitectureTab from './components/tabs/ArchitectureTab.svelte';
  import TrainingTab from './components/tabs/TrainingTab.svelte';
  import InferenceTab from './components/tabs/InferenceTab.svelte';
  import { ui, synthesis, applyPreset, type TabId } from './state.svelte';
  import { api } from './api';

  const TABS: { id: TabId; label: string }[] = [
    { id: 'orientation', label: 'Orientation' },
    { id: 'data', label: 'Data Synthesis' },
    { id: 'architecture', label: 'Model Architecture' },
    { id: 'training', label: 'Training' },
    { id: 'inference', label: 'Inference' },
  ];

  $effect(() => {
    (async () => {
      try {
        const { device } = await api.device();
        ui.device = device;
        ui.status = 'ready';
      } catch (e) {
        const msg = (e as Error).message || String(e);
        ui.status = `backend unreachable: ${msg} — is the server on :5041?`;
        console.error('initial fetch failed', e);
      }
    })();
  });

  // Load synthesis state at app start so every tab can rely on it. Without
  // this, going Orientation → Architecture → Training (skipping the Data
  // Synthesis tab) leaves synthesis.loaded = false, which breaks Training
  // (no symbols selected, batch can't load).
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
        console.error('synthesis preload failed', e);
      }
    })();
  });
</script>

<div class="h-screen w-screen flex flex-col overflow-hidden">
  <Header />

  <nav
    class="flex items-end gap-1 px-4 border-b border-[var(--color-border)]
           bg-[var(--color-surface)]/40 shrink-0"
  >
    {#each TABS as tab, i}
      <button
        type="button"
        class="tab-button"
        class:tab-button-active={ui.activeTab === tab.id}
        onclick={() => (ui.activeTab = tab.id)}
      >
        <span class="tab-number">{i}</span>
        <span>{tab.label}</span>
      </button>
    {/each}
  </nav>

  <main class="flex-1 min-h-0 overflow-auto">
    {#if ui.activeTab === 'orientation'}
      <OrientationTab />
    {:else if ui.activeTab === 'data'}
      <DataSynthesisTab />
    {:else if ui.activeTab === 'architecture'}
      <ArchitectureTab />
    {:else if ui.activeTab === 'training'}
      <TrainingTab />
    {:else if ui.activeTab === 'inference'}
      <InferenceTab />
    {/if}
  </main>

  <StatusBar />
</div>

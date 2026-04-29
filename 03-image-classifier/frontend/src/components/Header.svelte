<script lang="ts">
  import Icon from './Icon.svelte';
  import DeviceSelector from './DeviceSelector.svelte';
</script>

<header
  class="grid items-center gap-4 px-5 h-12 border-b border-[var(--color-border)] shrink-0
         bg-[var(--color-surface)]/60 backdrop-blur"
  style="grid-template-columns: auto 1fr auto"
>
  <!-- Left: logo + title -->
  <div class="flex items-center gap-2 min-w-0">
    <Icon name="sigma" size={18} class="text-[var(--color-accent)] shrink-0" />
    <span class="text-sm font-semibold tracking-tight truncate">Image Classifier</span>
  </div>

  <!-- Middle column intentionally empty (the previous "ready" status
       indicator was unused). -->
  <div></div>

  <!-- Right: device picker -->
  <DeviceSelector />
</header>

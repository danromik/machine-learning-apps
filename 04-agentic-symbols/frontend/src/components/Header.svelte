<script lang="ts">
  import Icon from './Icon.svelte';
  import DeviceSelector from './DeviceSelector.svelte';
  import { chat, persistChatPrefs } from '../state.svelte';

  function toggleChat() {
    chat.visible = !chat.visible;
    persistChatPrefs();
  }
</script>

<header
  class="grid items-center gap-4 px-5 h-12 border-b border-[var(--color-border)] shrink-0
         bg-[var(--color-surface)]/60 backdrop-blur"
  style="grid-template-columns: auto 1fr auto"
>
  <!-- Left: logo + title -->
  <div class="flex items-center gap-2 min-w-0">
    <Icon name="sigma" size={18} class="text-[var(--color-accent)] shrink-0" />
    <span class="text-sm font-semibold tracking-tight truncate">Agentic Symbol Trainer</span>
  </div>

  <!-- Middle column intentionally empty (the previous "ready" status
       indicator was unused). -->
  <div></div>

  <!-- Right: device picker + chat toggle -->
  <div class="flex items-center gap-2">
    <DeviceSelector />
    <button
      type="button"
      onclick={toggleChat}
      title={chat.visible ? 'Hide ML Engineer chat' : 'Show ML Engineer chat'}
      class="flex items-center gap-1.5 px-2.5 h-7 text-xs rounded-md border
             border-[var(--color-border)] bg-[var(--color-surface)]
             hover:bg-[var(--color-border)]/30 transition-colors"
      class:active={chat.visible}
    >
      <Icon name="message-square" size={13} />
      <span>ML Engineer</span>
    </button>
  </div>
</header>

<style>
  .active {
    background: color-mix(in srgb, var(--color-accent) 15%, transparent);
    border-color: var(--color-accent);
    color: var(--color-accent);
  }
</style>

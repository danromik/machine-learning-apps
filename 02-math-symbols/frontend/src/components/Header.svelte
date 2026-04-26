<script lang="ts">
  import Icon from './Icon.svelte';
  import { ui } from '../state.svelte';
</script>

<header
  class="grid items-center gap-4 px-5 h-12 border-b border-[var(--color-border)] shrink-0
         bg-[var(--color-surface)]/60 backdrop-blur"
  style="grid-template-columns: 1fr auto 1fr"
>
  <!-- Left: logo + title -->
  <div class="flex items-center gap-2 justify-self-start min-w-0">
    <Icon name="sigma" size={18} class="text-[var(--color-accent)] shrink-0" />
    <span class="text-sm font-semibold tracking-tight truncate">Math Symbol Trainer</span>
  </div>

  <!-- Center: status text -->
  <div class="flex items-center gap-2 justify-self-center min-w-0 max-w-full">
    <span class="text-sm truncate">{ui.status}</span>
  </div>

  <!-- Right: device -->
  <div class="flex items-center gap-5 text-xs text-[var(--color-muted)] justify-self-end min-w-0">
    <div class="flex items-center gap-1.5 shrink-0">
      <Icon name="cpu" size={14} />
      <span>device</span>
      <span class="font-mono text-[var(--color-text)]">{ui.device || '—'}</span>
    </div>
  </div>
</header>

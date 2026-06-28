<script lang="ts">
  import ThemeSelector from './ThemeSelector.svelte';
  import CubeStyleSelector from './CubeStyleSelector.svelte';
  import SoundToggle from './SoundToggle.svelte';
  import { training } from '../state.svelte';

  // Surface the background run from any tab, so a long/overnight run is always
  // visible. Falls back to the foreground status message when no run is active.
  let run = $derived(training.run);
  let dotClass = $derived.by(() => {
    const st = run?.state;
    if (st === 'running') return 'bg-emerald-500 animate-pulse';
    if (st === 'error') return 'bg-rose-500';
    if (st === 'finished' || st === 'stopped') return 'bg-[var(--color-muted)]';
    return '';
  });
</script>

<footer
  class="px-4 h-8 shrink-0 border-t border-[var(--color-border)]
         flex items-center justify-between gap-3
         bg-[var(--color-surface)]/80 backdrop-blur text-xs text-[var(--color-muted)]"
>
  <div class="flex items-center gap-2 min-w-0 font-mono truncate">
    {#if run && run.state !== 'idle'}
      <span class="inline-block w-2 h-2 rounded-full {dotClass}"></span>
      <span class="truncate">
        run {run.state} · iter {run.iteration} · k={run.current_k}{#if run.last_checkpoint}
          · ✓ {run.last_checkpoint}{/if}
      </span>
    {:else}
      <span class="truncate">{training.statusMsg}</span>
    {/if}
  </div>
  <div class="flex items-center gap-3 shrink-0">
    <SoundToggle />
    <CubeStyleSelector />
    <ThemeSelector />
  </div>
</footer>

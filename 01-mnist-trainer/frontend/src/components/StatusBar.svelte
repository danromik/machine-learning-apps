<script lang="ts">
  import ThemeSelector from './ThemeSelector.svelte';
</script>

<footer
  class="px-4 h-8 shrink-0 border-t border-[var(--color-border)]
         flex items-center justify-end
         bg-[var(--color-surface)]/80 backdrop-blur"
>
  <ThemeSelector />
</footer>

<script lang="ts">
  import { THEMES, applyTheme, type ThemeName } from '../theme';
  import { theme } from '../state.svelte';

  let current = $state<ThemeName>(
    (document.documentElement.getAttribute('data-theme') as ThemeName) ?? 'github-light',
  );

  function onChange(e: Event) {
    const id = (e.target as HTMLSelectElement).value as ThemeName;
    current = id;
    applyTheme(id);
    theme.version += 1;
  }
</script>

<label class="flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
  <span>Theme</span>
  <select
    class="input !py-0.5 !px-2 !text-xs !font-sans w-auto"
    value={current}
    onchange={onChange}
  >
    {#each THEMES as t}
      <option value={t.id}>{t.label}</option>
    {/each}
  </select>
</label>

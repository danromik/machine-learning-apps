<script lang="ts">
  import { CUBE_STYLES, persistCubeStyleId } from '../cubeStyles';
  import { cubeStyle } from '../state.svelte';

  function onChange(e: Event) {
    const id = (e.target as HTMLSelectElement).value;
    cubeStyle.id = id;
    persistCubeStyleId(id);
  }
</script>

<label class="flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
  <span>Cube</span>
  <select
    class="input !py-0.5 !px-2 !text-xs !font-sans w-auto"
    value={cubeStyle.id}
    onchange={onChange}
  >
    {#each CUBE_STYLES as s}
      <option value={s.id}>{s.label}</option>
    {/each}
  </select>
</label>

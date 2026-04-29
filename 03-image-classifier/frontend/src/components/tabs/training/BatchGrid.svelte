<script lang="ts">
  import { training } from '../../../state.svelte';

  let { onSelect }: { onSelect: (i: number) => void } = $props();
</script>

<div class="flex-1 min-w-0 min-h-0 overflow-auto p-2">
  {#if training.batch.length === 0}
    <div class="h-full flex items-center justify-center text-xs text-[var(--color-muted)]">
      Loading batch…
    </div>
  {:else}
    <div
      class="grid gap-1"
      style="grid-template-columns: repeat(auto-fill, minmax(48px, 1fr));"
    >
      {#each training.batch as sample, i (i)}
        {@const selected = training.selectedIndex === i}
        {@const verdict = training.batchVerdict[i] ?? null}
        {@const verdictColor = verdict === 'correct'
          ? 'var(--color-success)'
          : verdict === 'incorrect'
          ? 'var(--color-danger)'
          : null}
        {@const selectionColor = selected && training.animating
          ? 'var(--color-accent)'
          : verdictColor ?? 'var(--color-accent)'}
        <button
          type="button"
          class="relative aspect-square block p-0 rounded outline-none overflow-hidden"
          style="
            border: 2px solid {selected
              ? selectionColor
              : verdictColor ?? 'var(--color-border)'};
            background: var(--color-surface);
            box-shadow: {selected
              ? `0 0 0 3px ${selectionColor}, 0 6px 12px rgba(0,0,0,0.18)`
              : verdictColor
              ? `0 0 0 1px ${verdictColor}`
              : 'none'};
            transform: {selected ? 'scale(1.18)' : 'scale(1)'};
            z-index: {selected ? '10' : '0'};
            transition: transform 60ms ease-out, box-shadow 60ms, border-color 120ms;
          "
          onclick={() => onSelect(i)}
          title="{sample.label} · {sample.source}"
          aria-label={`Sample ${i + 1}: ${sample.label}`}
        >
          <img
            src="data:image/png;base64,{sample.png_b64}"
            alt={sample.label}
            class="w-full h-full block object-cover"
          />
        </button>
      {/each}
    </div>
  {/if}
</div>

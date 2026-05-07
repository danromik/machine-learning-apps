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
      style="grid-template-columns: repeat(auto-fill, minmax(40px, 1fr));"
    >
      {#each training.batch as sample, i (i)}
        {@const selected = training.selectedIndex === i}
        {@const verdict = training.batchVerdict[i] ?? null}
        {@const verdictColor = verdict === 'correct'
          ? 'var(--color-success)'
          : verdict === 'incorrect'
          ? 'var(--color-danger)'
          : null}
        <!-- The accent (blue) ring is reserved for the Train 1 Batch
             (Fun) sweep — it marks the sample currently being narrated.
             Clicks (or post-Fun selections) use the verdict color so the
             ring carries the same correctness signal as the cell border;
             we only fall back to accent if there's no verdict yet. -->
        {@const selectionColor = selected && training.animating
          ? 'var(--color-accent)'
          : verdictColor ?? 'var(--color-accent)'}
        <button
          type="button"
          class="relative aspect-square block p-0 rounded outline-none"
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
          title={sample.font + (sample.missing_glyph ? ' (missing glyph)' : '')}
          aria-label={`Sample ${i + 1}: ${sample.label}`}
        >
          <img
            src="data:image/png;base64,{sample.png_b64}"
            alt={sample.label}
            class="synth-img w-full h-full block"
          />
          {#if sample.missing_glyph}
            <span
              class="absolute top-0 right-0 px-1 text-[8px] leading-none
                     bg-[var(--color-danger)] text-white rounded-bl"
            >!</span>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>

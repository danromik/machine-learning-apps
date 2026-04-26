<script lang="ts">
  import Icon from '../Icon.svelte';
</script>

<div class="h-full w-full flex items-center justify-center p-8">
  <div class="card max-w-xl w-full p-8 flex flex-col items-center gap-3 text-center">
    <Icon name="type" size={32} class="text-[var(--color-accent)]" />
    <h2 class="text-lg font-semibold text-[var(--color-heading)]">Inference</h2>
    <p class="text-sm text-[var(--color-muted)] leading-relaxed">
      Type text and watch the trained classifier transcribe each glyph in real time,
      with top-K predictions and confidences for every character.
    </p>
    <span class="pill mt-2">empty placeholder</span>
  </div>
</div>

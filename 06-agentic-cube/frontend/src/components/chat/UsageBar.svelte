<!--
  Top of the chat pane: a header label on the left ("RL Coach Chat")
  and a context-usage indicator on the right showing how much of the 1M-
  token context the agent has consumed plus running cost in USD. The
  progress bar is context-window-aware: green at <50%, amber at 50-80%,
  red beyond.
-->
<script lang="ts">
  import { chat, CONTEXT_WINDOW } from '../../state.svelte';

  let usedTokens = $derived(
    chat.usage.input_tokens + chat.usage.output_tokens,
  );
  let pct = $derived(
    Math.min(100, (usedTokens / CONTEXT_WINDOW) * 100),
  );
  let barColor = $derived(
    pct < 50 ? 'var(--color-accent)' : pct < 80 ? '#d97706' : '#dc2626',
  );

  function fmtTokens(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k';
    return n.toString();
  }

  function fmtCost(c: number): string {
    if (c < 0.01) return '<$0.01';
    return '$' + c.toFixed(c < 1 ? 3 : 2);
  }
</script>

<div
  class="px-3 pt-2.5 pb-2 border-b border-[var(--color-border)] shrink-0
         flex items-center gap-3"
>
  <h2
    class="text-sm font-semibold text-[var(--color-heading)] truncate
           shrink-0"
  >
    RL Coach Chat
  </h2>
  <!-- Context indicator: label + tokens on top, progress bar below.
       Right-aligned in the remaining space so the header sits on the
       left and this stays as a compact gauge on the right. -->
  <div class="flex-1 min-w-0 ml-auto max-w-[200px]">
    <div
      class="flex items-center justify-between text-[10px] uppercase
             tracking-wider text-[var(--color-muted)] mb-1 gap-2"
    >
      <span>context</span>
      <span class="tabular-nums truncate">
        {fmtTokens(usedTokens)} / 1M · {fmtCost(chat.usage.total_cost_usd)}
      </span>
    </div>
    <div class="h-1.5 rounded-full bg-[var(--color-border)]/40 overflow-hidden">
      <div
        class="h-full rounded-full transition-[width] duration-300 ease-out"
        style="width: {pct}%; background: {barColor};"
      ></div>
    </div>
  </div>
</div>

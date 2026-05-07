<!--
  Renders the chat as bubbles + tool info rows. Tool calls and results
  live OUTSIDE bubbles (per spec); a regular message after a tool returns
  starts a new bubble.

  Assistant text is rendered as Markdown (via marked) so headings,
  lists, code blocks, and inline emphasis come through. Streaming
  re-parses on every delta — partial markdown still renders cleanly
  since marked tolerates open tokens.

  Tool calls show a friendly English label (e.g. "Setting hyperparameters")
  with an info icon — hover reveals a popover with the full input args
  and (once it returns) the raw result string.

  Auto-scrolls to bottom on every items change unless the user has
  manually scrolled away. Detected by comparing scroll position to
  scrollHeight; once "near bottom" (within 80px) is true, every new
  item snaps to the bottom again.
-->
<script lang="ts">
  import { marked } from 'marked';
  import { chat, type ChatItem } from '../../state.svelte';
  import { toolLabel } from './toolLabels';
  import Icon from '../Icon.svelte';

  // GFM gives tables / strikethrough / autolinks; breaks: true converts
  // single newlines to <br> which is what feels right in a chat (the
  // user expects "next line" without typing two newlines).
  marked.setOptions({ gfm: true, breaks: true });

  function renderMarkdown(text: string): string {
    try {
      // marked.parse can be sync or async depending on extensions; with
      // our default config it's sync, so cast is safe.
      return marked.parse(text) as string;
    } catch {
      // Malformed input — fall back to escaped plain text rather than
      // tearing down the bubble.
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }
  }

  let scroller: HTMLDivElement | null = $state(null);
  // True when the user is pinned at (or near) the bottom — new content
  // should keep them there. False when they've scrolled up to read
  // history; we leave their viewport alone.
  let pinToBottom = $state(true);

  function onScroll() {
    if (!scroller) return;
    const distance =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    pinToBottom = distance < 80;
  }

  // Re-pin to the bottom whenever the items array length grows OR the
  // last item's text length grows (streaming tokens into the active
  // assistant bubble). Reading the relevant fields registers them as
  // dependencies for $effect.
  $effect(() => {
    const last = chat.items[chat.items.length - 1];
    void chat.items.length;
    if (last && last.kind === 'text') void last.text;
    if (last && last.kind === 'tool') void (last.result?.length ?? 0);
    if (pinToBottom && scroller) {
      // Defer to after the DOM settles so we measure the post-render
      // scrollHeight, not the pre-render one.
      queueMicrotask(() => {
        if (scroller) scroller.scrollTop = scroller.scrollHeight;
      });
    }
  });

  function fmtJson(v: unknown): string {
    if (v === null || v === undefined) return '';
    try {
      return JSON.stringify(v, null, 2);
    } catch {
      return String(v);
    }
  }

  // The popover shows the full result; the chat row stays clean.
  // 4KB is plenty of context for the user to inspect the call without
  // forcing the popover to scroll across the whole pane.
  const POPOVER_RESULT_LIMIT = 4000;

  function fmtResult(item: Extract<ChatItem, { kind: 'tool' }>): string {
    if (!item.result) return '(no result yet)';
    if (item.result.length > POPOVER_RESULT_LIMIT) {
      return item.result.slice(0, POPOVER_RESULT_LIMIT) + `\n… (+${item.result.length - POPOVER_RESULT_LIMIT} more chars)`;
    }
    return item.result;
  }
</script>

<div
  bind:this={scroller}
  onscroll={onScroll}
  class="flex-1 min-h-0 overflow-y-auto px-3 py-3 flex flex-col gap-2"
>
  {#each chat.items as item, i (i)}
    {#if item.kind === 'text'}
      <div
        class="max-w-[85%] px-3 py-2 rounded-lg text-sm leading-relaxed"
        class:self-end={item.role === 'user'}
        class:self-start={item.role === 'assistant'}
        class:bubble-user={item.role === 'user'}
        class:bubble-assistant={item.role === 'assistant'}
      >
        {#if item.role === 'assistant'}
          <span class="md-content">{@html renderMarkdown(item.text)}</span>{#if item.streaming}<span class="streaming-cursor"></span>{/if}
        {:else}
          <span class="whitespace-pre-wrap break-words">{item.text}</span>
        {/if}
      </div>
    {:else if item.kind === 'tool'}
      <div
        class="self-start text-[11px] text-[var(--color-muted)] flex items-center
               gap-1.5 max-w-full tool-row"
      >
        <span class="shrink-0">
          {#if item.status === 'running'}
            <span class="inline-block w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse"></span>
          {:else if item.status === 'success'}
            <Icon name="check" size={12} class="text-emerald-500" />
          {:else}
            <Icon name="x" size={12} class="text-rose-500" />
          {/if}
        </span>
        <span class="text-[var(--color-text)]">{toolLabel(item.name, item.input)}</span>
        <span class="info-wrapper relative inline-flex items-center">
          <button
            type="button"
            tabindex="-1"
            class="text-[var(--color-muted)]/70 hover:text-[var(--color-accent)] transition-colors
                   inline-flex items-center cursor-help"
            aria-label="Show details"
          >
            <Icon name="info" size={11} />
          </button>
          <span
            class="info-popover absolute left-0 top-full mt-1 z-20 hidden
                   min-w-[220px] max-w-[340px] rounded-md border
                   border-[var(--color-border)] bg-[var(--color-surface)]
                   shadow-xl p-2 text-[10px] leading-snug"
          >
            <div class="text-[var(--color-muted)] uppercase tracking-wider mb-0.5 text-[9px]">
              {item.name}
            </div>
            <div class="text-[var(--color-muted)] uppercase tracking-wider mt-1.5 mb-0.5 text-[9px]">
              input
            </div>
            <pre
              class="font-mono text-[var(--color-text)] whitespace-pre-wrap break-all
                     max-h-[180px] overflow-auto bg-[var(--color-bg)]/40 rounded p-1.5"
            >{fmtJson(item.input) || '{}'}</pre>
            <div class="text-[var(--color-muted)] uppercase tracking-wider mt-1.5 mb-0.5 text-[9px]">
              result
            </div>
            <pre
              class="font-mono text-[var(--color-text)] whitespace-pre-wrap break-all
                     max-h-[260px] overflow-auto bg-[var(--color-bg)]/40 rounded p-1.5"
              class:text-rose-500={item.status === 'error'}
            >{fmtResult(item)}</pre>
          </span>
        </span>
      </div>
    {:else if item.kind === 'error'}
      <div
        class="self-stretch text-xs text-rose-500 bg-rose-500/10 border
               border-rose-500/30 rounded px-3 py-2"
      >
        {item.message}
      </div>
    {/if}
  {/each}

  {#if chat.items.length === 0}
    <div
      class="m-auto text-center text-[var(--color-muted)] text-sm leading-relaxed
             px-6 py-8 max-w-xs"
    >
      <Icon name="brain" size={24} class="mx-auto mb-3 opacity-60" />
      <p class="mb-2 text-[var(--color-heading)] font-medium">ML Engineer</p>
      <p>
        Ask for help understanding the pipeline, or say "train this for me"
        and I'll set up and run the model end-to-end.
      </p>
    </div>
  {/if}
</div>

<style>
  .bubble-user {
    background: var(--color-accent);
    color: white;
  }
  .bubble-assistant {
    background: var(--color-border);
    color: var(--color-text);
  }
  /* Brighten on dark themes — color-mix lets us blend without naming
     the theme; works against any accent. */
  :global([data-theme='one-dark']) .bubble-assistant {
    background: color-mix(in srgb, var(--color-border) 60%, transparent);
  }

  /* Streaming cursor — a soft pulsing bar that sits at the end of the
     bubble while tokens are still arriving. */
  .streaming-cursor {
    display: inline-block;
    width: 2px;
    height: 1em;
    margin-left: 1px;
    background: currentColor;
    opacity: 0.4;
    vertical-align: text-bottom;
    animation: streaming-pulse 1s ease-in-out infinite;
  }
  @keyframes streaming-pulse {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 0.6; }
  }

  /* ── Markdown content styling ────────────────────────────────────
     Override default browser margins so paragraphs / lists / code
     don't blow out the bubble. Keep just enough vertical rhythm to
     separate adjacent blocks.
  */
  .md-content :global(p) { margin: 0; }
  .md-content :global(p + p),
  .md-content :global(ul + p),
  .md-content :global(ol + p),
  .md-content :global(p + ul),
  .md-content :global(p + ol),
  .md-content :global(p + pre),
  .md-content :global(pre + p),
  .md-content :global(pre + pre) { margin-top: 0.5em; }

  .md-content :global(ul),
  .md-content :global(ol) {
    margin: 0;
    padding-left: 1.25em;
  }
  .md-content :global(li) { margin: 0.1em 0; }
  .md-content :global(li > p) { margin: 0; }

  .md-content :global(h1),
  .md-content :global(h2),
  .md-content :global(h3),
  .md-content :global(h4) {
    margin: 0.6em 0 0.2em;
    font-weight: 600;
    line-height: 1.25;
  }
  .md-content :global(h1) { font-size: 1.1em; }
  .md-content :global(h2) { font-size: 1.05em; }
  .md-content :global(h3) { font-size: 1em; }
  .md-content :global(h4) { font-size: 0.95em; }

  .md-content :global(code) {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    font-size: 0.85em;
    background: color-mix(in srgb, var(--color-text) 10%, transparent);
    padding: 0.1em 0.3em;
    border-radius: 3px;
  }
  .md-content :global(pre) {
    background: color-mix(in srgb, var(--color-text) 8%, transparent);
    border-radius: 4px;
    padding: 0.5em 0.7em;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.4;
  }
  .md-content :global(pre code) {
    background: transparent;
    padding: 0;
    border-radius: 0;
    font-size: inherit;
  }
  .md-content :global(a) {
    color: var(--color-accent);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .md-content :global(strong) { font-weight: 600; }
  .md-content :global(em) { font-style: italic; }
  .md-content :global(blockquote) {
    margin: 0.4em 0;
    padding-left: 0.7em;
    border-left: 2px solid color-mix(in srgb, var(--color-text) 25%, transparent);
    color: var(--color-muted);
  }
  .md-content :global(hr) {
    margin: 0.6em 0;
    border: 0;
    border-top: 1px solid color-mix(in srgb, var(--color-text) 20%, transparent);
  }
  .md-content :global(table) {
    border-collapse: collapse;
    margin: 0.4em 0;
  }
  .md-content :global(th),
  .md-content :global(td) {
    border: 1px solid color-mix(in srgb, var(--color-text) 20%, transparent);
    padding: 0.2em 0.5em;
    text-align: left;
  }

  /* Inside a user bubble (white-on-accent) inline code/links need
     contrast adjustments so they don't disappear. */
  .bubble-user :global(code) {
    background: rgba(255, 255, 255, 0.18);
  }
  .bubble-user :global(a) {
    color: white;
    text-decoration: underline;
  }

  /* ── Tool-call info popover ─────────────────────────────────────── */
  .info-wrapper:hover .info-popover,
  .info-wrapper:focus-within .info-popover {
    display: block;
  }
  /* When the row sits near the bottom of the chat, the popover (which
     would otherwise expand below) gets clipped. Allow horizontal
     overflow on the row so the popover can use its full width. */
  .tool-row { overflow: visible; }
</style>

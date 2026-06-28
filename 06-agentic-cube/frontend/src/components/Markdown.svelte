<!--
  Renders a markdown string to styled HTML via `marked`. Shared by the chat
  transcript and the RL-Coach-written reports (Progress Report + Debrief tabs).
  Note: `{@html}` is unsanitized — content comes from our own model output.
-->
<script lang="ts">
  import { marked } from 'marked';

  let { source = '' }: { source?: string } = $props();

  marked.setOptions({ gfm: true, breaks: true });

  function render(text: string): string {
    try {
      return marked.parse(text) as string;
    } catch {
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  }
</script>

<div class="md-content">{@html render(source)}</div>

<style>
  .md-content :global(p) { margin: 0; }
  .md-content :global(p + p),
  .md-content :global(ul + p),
  .md-content :global(ol + p),
  .md-content :global(p + ul),
  .md-content :global(p + ol),
  .md-content :global(p + pre),
  .md-content :global(pre + p),
  .md-content :global(h1 + p),
  .md-content :global(h2 + p),
  .md-content :global(h3 + p),
  .md-content :global(pre + pre) { margin-top: 0.6em; }

  .md-content :global(ul),
  .md-content :global(ol) { margin: 0.2em 0; padding-left: 1.3em; }
  .md-content :global(li) { margin: 0.15em 0; }
  .md-content :global(li > p) { margin: 0; }

  .md-content :global(h1),
  .md-content :global(h2),
  .md-content :global(h3),
  .md-content :global(h4) {
    margin: 0.9em 0 0.3em;
    font-weight: 600;
    line-height: 1.25;
    color: var(--color-heading);
  }
  .md-content :global(h1) { font-size: 1.35em; }
  .md-content :global(h2) { font-size: 1.18em; }
  .md-content :global(h3) { font-size: 1.05em; }
  .md-content :global(h4) { font-size: 0.98em; }

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
    padding: 0.6em 0.8em;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.4;
  }
  .md-content :global(pre code) { background: transparent; padding: 0; border-radius: 0; }
  .md-content :global(a) {
    color: var(--color-accent);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .md-content :global(strong) { font-weight: 600; color: var(--color-heading); }
  .md-content :global(em) { font-style: italic; }
  .md-content :global(blockquote) {
    margin: 0.5em 0;
    padding-left: 0.8em;
    border-left: 3px solid color-mix(in srgb, var(--color-accent) 50%, transparent);
    color: var(--color-muted);
  }
  .md-content :global(hr) {
    margin: 0.9em 0;
    border: 0;
    border-top: 1px solid color-mix(in srgb, var(--color-text) 18%, transparent);
  }
  .md-content :global(table) { border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }
  .md-content :global(th),
  .md-content :global(td) {
    border: 1px solid color-mix(in srgb, var(--color-text) 18%, transparent);
    padding: 0.25em 0.6em;
    text-align: left;
  }
</style>

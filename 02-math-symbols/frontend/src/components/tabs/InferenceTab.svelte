<script lang="ts">
  import { onMount } from 'svelte';
  import katex from 'katex';
  import { api, type InferenceItem } from '../../api';
  import { synthesis, training } from '../../state.svelte';

  // Default starter — illustrates the plain-text + $math$ split.
  let inputText = $state(
    'Plug into the formula: $f(x) = x^2 + \\alpha \\cdot \\beta$'
  );
  let glyphs = $state<string[]>([]);
  let items = $state<InferenceItem[]>([]);
  let backendHasSession = $state(false);
  let busy = $state(false);
  let errorMsg = $state<string | null>(null);

  type Segment =
    | { kind: 'text'; content: string }
    | { kind: 'math'; content: string };

  // Split the input into alternating plain-text and $…$ math segments.
  // An unclosed `$` swallows the rest of the input as math (KaTeX will
  // either render what it can or surface a friendly error inline).
  function parseSegments(input: string): Segment[] {
    const segments: Segment[] = [];
    let i = 0;
    while (i < input.length) {
      if (input[i] === '$') {
        const end = input.indexOf('$', i + 1);
        if (end === -1) {
          segments.push({ kind: 'math', content: input.slice(i + 1) });
          break;
        }
        segments.push({ kind: 'math', content: input.slice(i + 1, end) });
        i = end + 1;
      } else {
        const next = input.indexOf('$', i);
        const stop = next === -1 ? input.length : next;
        segments.push({ kind: 'text', content: input.slice(i, stop) });
        i = stop;
      }
    }
    return segments;
  }

  let segments = $derived(parseSegments(inputText));

  function escapeHtml(s: string): string {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // Build the preview HTML by rendering each segment in turn. Math
  // segments use KaTeX inline mode so they flow inline with surrounding
  // text. Wrap text segments in a `prose-text` span so the CSS can
  // restore the normal body font (KaTeX's CSS resets things aggressively).
  let previewHtml = $derived.by(() => {
    let html = '';
    for (const seg of segments) {
      if (seg.kind === 'text') {
        html += `<span class="prose-text">${escapeHtml(seg.content)}</span>`;
      } else {
        try {
          html += katex.renderToString(seg.content, {
            throwOnError: false,
            displayMode: false,
            output: 'html',
          });
        } catch (e) {
          html += `<span style="color:var(--color-danger)">${escapeHtml(
            (e as Error).message
          )}</span>`;
        }
      }
    }
    return html;
  });

  // Training fonts the user picked, in priority order — passed to the
  // backend's render endpoint so inference uses the same font pool the
  // model was trained on, with a graceful per-glyph fallback.
  let trainingFonts = $derived.by(() => {
    if (!synthesis.loaded) return [] as string[];
    return synthesis.fonts
      .filter((f) => synthesis.fontUsage[f.family] === 'train')
      .map((f) => f.family);
  });

  // Walk a KaTeX-rendered DOM tree and pull out the sequence of glyphs
  // in reading order. Whitespace and KaTeX's invisible spacing artifacts
  // are skipped so the grid only contains characters that could plausibly
  // be classified.
  function extractGlyphs(html: string): string[] {
    if (!html) return [];
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    const out: string[] = [];
    const walker = document.createTreeWalker(tmp, NodeFilter.SHOW_TEXT);
    let node: Node | null = walker.nextNode();
    while (node) {
      const text = node.textContent ?? '';
      for (const ch of text) {
        if (/\s/.test(ch)) continue;
        out.push(ch);
      }
      node = walker.nextNode();
    }
    return out;
  }

  // Initial run on mount so the user sees output without having to click
  // immediately. Subsequent runs are triggered by the Run Inference button.
  onMount(() => {
    runInference();
  });

  let inferenceSeq = 0;
  async function runInference() {
    const seq = ++inferenceSeq;
    const html = previewHtml;
    const extracted = extractGlyphs(html);
    glyphs = extracted;

    if (extracted.length === 0) {
      items = [];
      errorMsg = null;
      return;
    }
    busy = true;
    errorMsg = null;
    try {
      const fonts = trainingFonts.length > 0 ? trainingFonts : undefined;
      const result = await api.inferenceRender(extracted, fonts);
      if (seq !== inferenceSeq) return;
      items = result.items;
      backendHasSession = result.has_session;
    } catch (e) {
      if (seq !== inferenceSeq) return;
      errorMsg = (e as Error).message;
    } finally {
      if (seq === inferenceSeq) busy = false;
    }
  }

  let predictedText = $derived(
    items.map((it) => it.predicted_char ?? '·').join('')
  );
</script>

<div class="h-full overflow-auto">
  <div class="max-w-4xl mx-auto px-4 py-3 flex flex-col gap-2">
    <!-- ── Input ─────────────────────────────────────────────────────── -->
    <section class="card p-3 flex flex-col gap-2">
      <header class="flex items-baseline justify-between gap-3 flex-wrap">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Input
        </h3>
        <span class="text-xs text-[var(--color-muted)] font-mono">
          plain text + <code>$math$</code> (e.g. <code>$\alpha + \beta$</code>)
        </span>
      </header>
      <textarea
        class="input !text-sm !font-mono w-full resize-y"
        rows="3"
        spellcheck="false"
        autocomplete="off"
        placeholder="Type plain text. Wrap math in $…$"
        value={inputText}
        oninput={(e) =>
          (inputText = (e.currentTarget as HTMLTextAreaElement).value)}
      ></textarea>
      <div class="flex items-center justify-between gap-3 flex-wrap">
        <button
          type="button"
          class="btn-primary text-xs"
          onclick={runInference}
          disabled={busy || !inputText.trim()}
        >
          {busy ? 'Running…' : 'Run Inference'}
        </button>
        {#if !synthesis.loaded || trainingFonts.length === 0}
          <span class="text-[11px] text-[var(--color-muted)]">
            Tip: pick training fonts in the Data Synthesis tab so inference
            renders glyphs in the same fonts the model trained on.
          </span>
        {/if}
      </div>
    </section>

    <hr class="border-[var(--color-border)]" />

    <!-- ── KaTeX preview ─────────────────────────────────────────────── -->
    <section class="card p-4 flex flex-col gap-2">
      <header class="flex items-baseline justify-between gap-3">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Rendered preview
        </h3>
      </header>
      <div class="inference-preview min-h-12 py-2 leading-relaxed">
        {#if previewHtml}
          <!-- KaTeX output is sanitized; we render it directly. -->
          <!-- eslint-disable-next-line svelte/no-at-html-tags -->
          {@html previewHtml}
        {:else}
          <span class="text-xs text-[var(--color-muted)]">— empty —</span>
        {/if}
      </div>
    </section>

    <hr class="border-[var(--color-border)]" />

    <!-- ── Original glyph grid ──────────────────────────────────────── -->
    <section class="card p-3 flex flex-col gap-2">
      <header class="flex items-baseline justify-between gap-3 flex-wrap">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Glyphs
        </h3>
        <span class="text-xs text-[var(--color-muted)] font-mono">
          {glyphs.length} glyph{glyphs.length === 1 ? '' : 's'} extracted
        </span>
      </header>
      {#if glyphs.length === 0}
        <div class="text-xs text-[var(--color-muted)] py-2">— nothing to render —</div>
      {:else}
        <div class="inference-glyph-grid">
          {#each items as item, i (i)}
            <div
              class="inference-cell"
              title={`'${item.char}' · ${item.input_font ?? 'no supporting font'}`}
            >
              {#if item.input_png_b64}
                <img
                  src="data:image/png;base64,{item.input_png_b64}"
                  alt={item.char}
                  class="synth-img inference-img"
                />
              {:else}
                <div class="inference-img inference-img-missing">?</div>
              {/if}
              <span class="inference-cell-label">{item.char}</span>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <hr class="border-[var(--color-border)]" />

    <!-- ── Predicted glyph grid ──────────────────────────────────────── -->
    <section class="card p-3 flex flex-col gap-2">
      <header class="flex items-baseline justify-between gap-3 flex-wrap">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Predicted glyphs
        </h3>
        <span class="text-xs text-[var(--color-muted)] font-mono">
          {#if !backendHasSession && !busy && items.length > 0}
            no training session — initialize a model first
          {:else if busy}
            classifying…
          {:else if items.length > 0}
            top-1 prediction per glyph
          {/if}
        </span>
      </header>
      {#if items.length === 0}
        <div class="text-xs text-[var(--color-muted)] py-2">— nothing to predict —</div>
      {:else if !backendHasSession}
        <div class="text-xs text-[var(--color-muted)] py-2">
          Initialize a model in the Training tab to see predictions.
        </div>
      {:else}
        <div class="inference-glyph-grid">
          {#each items as item, i (i)}
            {@const correct =
              item.predicted_char !== null && item.predicted_char === item.char}
            {@const out_of_class =
              item.predicted_char !== null && !item.in_class_set}
            <div
              class="inference-cell"
              class:inference-cell-correct={correct}
              class:inference-cell-incorrect={item.predicted_char !== null && !correct && !out_of_class}
              title={item.predicted_char === null
                ? '(no prediction)'
                : `predicted '${item.predicted_char}' · ${
                    item.confidence !== null
                      ? (item.confidence * 100).toFixed(1) + '%'
                      : ''
                  }${out_of_class ? ' · input not in class set' : ''}`}
            >
              {#if item.predicted_png_b64}
                <img
                  src="data:image/png;base64,{item.predicted_png_b64}"
                  alt={item.predicted_char ?? ''}
                  class="synth-img inference-img"
                />
              {:else}
                <div class="inference-img inference-img-missing">?</div>
              {/if}
              <span class="inference-cell-label">
                {item.predicted_char ?? '·'}
              </span>
            </div>
          {/each}
        </div>
      {/if}
    </section>

    <hr class="border-[var(--color-border)]" />

    <!-- ── Predicted text ────────────────────────────────────────────── -->
    <section class="card p-3 flex flex-col gap-1">
      <header class="flex items-baseline justify-between gap-3">
        <h3 class="text-sm font-semibold text-[var(--color-heading)]">
          Predicted text
        </h3>
        {#if backendHasSession && items.length > 0}
          <span class="text-xs text-[var(--color-muted)] font-mono">
            step {training.step.toLocaleString()} · {training.numClasses} classes
          </span>
        {/if}
      </header>
      <div class="font-mono text-base text-[var(--color-text)] py-2 break-all">
        {#if items.length === 0}
          <span class="text-[var(--color-muted)] text-sm">—</span>
        {:else if !backendHasSession}
          <span class="text-[var(--color-muted)] text-sm">
            (no session)
          </span>
        {:else}
          {predictedText}
        {/if}
      </div>
    </section>

    {#if errorMsg}
      <div class="text-xs text-[var(--color-danger)]">{errorMsg}</div>
    {/if}
  </div>
</div>

<style>
  /* Plain-text spans inside the preview need to be visually distinct
     from inline KaTeX output (KaTeX uses its own font stack). Use the
     normal body font + line height so the prose flows naturally. */
  .inference-preview :global(.prose-text) {
    font-family: var(--font-sans);
    color: var(--color-text);
    white-space: pre-wrap;
  }
  .inference-glyph-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(56px, 1fr));
    gap: 0.5rem;
  }
  .inference-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.125rem;
    padding: 0.25rem;
    border: 1px solid var(--color-border);
    border-radius: 0.375rem;
    background: var(--color-surface);
  }
  .inference-cell-correct {
    border-color: var(--color-success);
    box-shadow: 0 0 0 1px var(--color-success);
  }
  .inference-cell-incorrect {
    border-color: var(--color-danger);
    box-shadow: 0 0 0 1px var(--color-danger);
  }
  .inference-img {
    width: 3rem;
    height: 3rem;
    display: block;
  }
  .inference-img-missing {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-muted);
    font-family: var(--font-mono);
    font-size: 1.25rem;
    border: 1px dashed var(--color-border);
    border-radius: 0.25rem;
  }
  .inference-cell-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--color-muted);
    line-height: 1;
  }
</style>

<script lang="ts">
  import { onMount } from 'svelte';
  import katex from 'katex';
  import { api, type InferenceItem } from '../../api';
  import { synthesis } from '../../state.svelte';

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
    // Clear the detail panel — its currently-shown cell is from the
    // previous run and won't line up with the new items array.
    hoveredIdx = null;
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

  // The detail panel populates with whatever cell the user last hovered
  // on the Predicted glyphs grid. We never auto-clear on mouseleave so
  // the panel stays readable while the user moves the pointer over to
  // it; clicking Run Inference resets it (handled in runInference).
  let hoveredIdx = $state<number | null>(null);
  let hoveredItem = $derived(
    hoveredIdx !== null && hoveredIdx >= 0 && hoveredIdx < items.length
      ? items[hoveredIdx]
      : null
  );
</script>

<div class="h-full overflow-auto">
  <div class="max-w-4xl mx-auto px-4 py-3 flex flex-col gap-2">
    <!-- ── Input ─────────────────────────────────────────────────────── -->
    <section class="card p-3 flex flex-col gap-2">
      <label class="flex items-center gap-2">
        <span class="text-sm font-semibold text-[var(--color-heading)] shrink-0"
          >Input:</span>
        <input
          type="text"
          class="input !text-sm !font-mono w-full"
          spellcheck="false"
          autocomplete="off"
          value={inputText}
          oninput={(e) =>
            (inputText = (e.currentTarget as HTMLInputElement).value)}
        />
      </label>
      <div class="flex items-center justify-between gap-3 flex-wrap">
        <span class="text-xs text-[var(--color-muted)] font-mono">
          enter plain text + <code>$math$</code>
        </span>
        <button
          type="button"
          class="btn-primary text-xs"
          onclick={runInference}
          disabled={busy || !inputText.trim()}
        >
          {busy ? 'Running…' : 'Run Inference'}
        </button>
      </div>
    </section>

    <hr class="border-[var(--color-border)]" />

    <!-- ── KaTeX preview ─────────────────────────────────────────────── -->
    <section class="card p-3 flex items-baseline gap-3">
      <span class="text-sm font-semibold text-[var(--color-heading)] shrink-0"
        >Rendered preview:</span>
      <div class="inference-preview leading-relaxed flex-1 min-w-0">
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

    <!-- Predicted glyphs + Predicted text live in a narrower left
         column; the per-glyph detail panel sits in a sibling column
         spanning only row 1, so its bottom aligns with the predicted
         glyphs section's bottom — predicted text drops directly under
         the glyphs section in row 2. -->
    <div class="inference-bottom-grid">
      <!-- ── Predicted glyph grid (row 1, col 1) ────────────────── -->
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
                top-1 prediction per glyph · hover for details
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
                  class:inference-cell-hovered={hoveredIdx === i}
                  onmouseenter={() => (hoveredIdx = i)}
                  role="button"
                  tabindex="0"
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
                  <!-- Confidence "progress bar" for the top-1 prediction.
                       Width tracks confidence directly: empty = 0%, full
                       = 100%. The cell border already conveys correctness;
                       this row only conveys how sure the model was. -->
                  <div class="inference-confidence">
                    <div
                      class="inference-confidence-fill"
                      style="width: {((item.confidence ?? 0) * 100).toFixed(1)}%;"
                    ></div>
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </section>

      <!-- ── Per-glyph detail panel (row 1, col 2) ──────────────── -->
      <aside class="card p-3 flex flex-col gap-2 inference-detail-panel">
        <header class="flex items-baseline justify-between gap-2">
          <h3 class="text-sm font-semibold text-[var(--color-heading)]">
            Glyph detail
          </h3>
          {#if hoveredItem && hoveredIdx !== null}
            <span class="text-[10px] text-[var(--color-muted)] font-mono">
              #{hoveredIdx + 1}
            </span>
          {/if}
        </header>

        {#if !hoveredItem}
          <div class="text-xs text-[var(--color-muted)] py-2 leading-relaxed">
            Hover a predicted glyph to see top-K alternatives and confidences.
          </div>
        {:else}
          <!-- Input glyph + label -->
          <div class="flex items-center gap-2">
            {#if hoveredItem.input_png_b64}
              <img
                src="data:image/png;base64,{hoveredItem.input_png_b64}"
                alt={hoveredItem.char}
                class="synth-img inference-detail-input"
              />
            {:else}
              <div class="inference-detail-input inference-img-missing">?</div>
            {/if}
            <div class="flex flex-col text-xs leading-tight">
              <span class="text-[10px] text-[var(--color-muted)] uppercase tracking-wide">
                input
              </span>
              <span class="font-mono text-sm text-[var(--color-text)]">
                '{hoveredItem.char}'
              </span>
              {#if hoveredItem.input_font}
                <span class="text-[10px] text-[var(--color-muted)] truncate">
                  {hoveredItem.input_font}
                </span>
              {/if}
            </div>
          </div>

          <!-- Top-1 confidence value — same number that drives the
               progress bar in each predicted-glyphs cell, surfaced
               here as text so the user can read the actual percentage. -->
          {#if hoveredItem.confidence !== null}
            <div
              class="flex items-baseline justify-between text-xs
                     border-t border-[var(--color-border)] pt-1.5"
            >
              <span
                class="text-[10px] text-[var(--color-muted)] uppercase tracking-wide"
              >
                top-1 confidence
              </span>
              <span class="font-mono tabular-nums text-[var(--color-text)]">
                {(hoveredItem.confidence * 100).toFixed(1)}%
              </span>
            </div>
          {/if}

          <!-- Top-K alternatives -->
          {#if hoveredItem.top_k.length === 0}
            <div class="text-xs text-[var(--color-muted)] py-1">
              {backendHasSession ? '— no alternatives —' : 'no session'}
            </div>
          {:else}
            <div class="flex flex-col gap-1">
              <span
                class="text-[10px] text-[var(--color-muted)] uppercase tracking-wide"
              >
                top {hoveredItem.top_k.length} predictions
              </span>
              {#each hoveredItem.top_k as alt, k (k)}
                {@const isInputClass = alt.char === hoveredItem.char}
                <div class="inference-detail-row">
                  <span class="inference-detail-rank">{k + 1}</span>
                  {#if alt.png_b64}
                    <img
                      src="data:image/png;base64,{alt.png_b64}"
                      alt={alt.char}
                      class="synth-img inference-detail-altimg"
                    />
                  {:else}
                    <div class="inference-detail-altimg inference-img-missing">?</div>
                  {/if}
                  <span
                    class="inference-detail-altchar"
                    class:inference-detail-altchar-match={isInputClass}
                    >{alt.char}</span>
                  <div class="inference-confidence inference-detail-bar">
                    <div
                      class="inference-confidence-fill"
                      style="
                        width: {(alt.confidence * 100).toFixed(1)}%;
                        background: {isInputClass
                          ? 'var(--color-success)'
                          : k === 0
                          ? 'var(--color-danger)'
                          : 'var(--color-accent)'};
                      "
                    ></div>
                  </div>
                  <span
                    class="inference-detail-pct font-mono tabular-nums"
                    >{(alt.confidence * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
          {/if}
        {/if}
      </aside>

      <!-- ── Predicted text (row 2, col 1; col 2 stays empty so the
           panel's bottom edge aligns with the predicted glyphs section
           above, not with this text row) ─────────────────────────── -->
      <section
        class="card p-3 flex items-baseline gap-3 flex-wrap inference-text-row"
      >
        <span class="text-sm font-semibold text-[var(--color-heading)] shrink-0"
          >Predicted text:</span>
        <span
          class="font-mono text-base text-[var(--color-text)] break-all flex-1 min-w-0"
        >
          {#if items.length === 0}
            <span class="text-[var(--color-muted)] text-sm">—</span>
          {:else if !backendHasSession}
            <span class="text-[var(--color-muted)] text-sm">(no session)</span>
          {:else}
            {predictedText}
          {/if}
        </span>
      </section>
    </div>

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
    grid-template-columns: repeat(auto-fill, minmax(36px, 1fr));
    gap: 0.25rem;
  }
  .inference-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.0625rem;
    padding: 0.125rem;
    border: 1px solid var(--color-border);
    border-radius: 0.25rem;
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
  /* Hovered cell — distinct accent ring layered ON TOP of any verdict
     ring so the user can still tell whether the prediction was right. */
  .inference-cell-hovered {
    box-shadow:
      0 0 0 2px var(--color-accent),
      0 4px 10px rgba(0, 0, 0, 0.15);
    transform: translateY(-1px);
  }
  .inference-cell {
    transition:
      box-shadow 80ms,
      transform 80ms;
  }
  /* Per-cell confidence bar — sits at the bottom of each cell, fills
     left-to-right as the top-1 confidence rises. Empty = 0, full = 100. */
  .inference-confidence {
    width: 100%;
    height: 3px;
    background: var(--color-surface-2);
    border-radius: 2px;
    overflow: hidden;
  }
  .inference-confidence-fill {
    height: 100%;
    background: var(--color-accent);
    transition: width 120ms ease-out;
  }
  .inference-img {
    width: 2rem;
    height: 2rem;
    display: block;
  }
  .inference-img-missing {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--color-muted);
    font-family: var(--font-mono);
    font-size: 0.875rem;
    border: 1px dashed var(--color-border);
    border-radius: 0.2rem;
  }
  .inference-cell-label {
    font-family: var(--font-mono);
    font-size: 0.625rem;
    color: var(--color-muted);
    line-height: 1;
  }

  /* ── Bottom grid: predicted glyphs (row 1, col 1) and predicted
        text (row 2, col 1) sit in the left column; the detail panel
        spans both rows in the right column so its bottom edge sits at
        the predicted text section's bottom edge. ──────────────────── */
  .inference-bottom-grid {
    display: grid;
    grid-template-columns: 1fr 16rem;
    grid-template-rows: auto auto;
    gap: 0.5rem;
  }
  .inference-text-row {
    grid-column: 1 / 2;
    grid-row: 2 / 3;
  }
  .inference-detail-panel {
    grid-column: 2 / 3;
    grid-row: 1 / 3;
  }
  .inference-detail-input {
    width: 2.5rem;
    height: 2.5rem;
    display: block;
  }
  .inference-detail-row {
    display: grid;
    grid-template-columns: 0.75rem 1.25rem 1rem 1fr 2.5rem;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.7rem;
  }
  .inference-detail-rank {
    color: var(--color-muted);
    font-family: var(--font-mono);
    font-size: 0.625rem;
    text-align: right;
  }
  .inference-detail-altimg {
    width: 1.25rem;
    height: 1.25rem;
    display: block;
  }
  .inference-detail-altchar {
    font-family: var(--font-mono);
    color: var(--color-text);
    text-align: center;
  }
  .inference-detail-altchar-match {
    color: var(--color-success);
    font-weight: 600;
  }
  .inference-detail-bar {
    height: 6px;
  }
  .inference-detail-pct {
    color: var(--color-muted);
    font-size: 0.625rem;
    text-align: right;
  }
</style>

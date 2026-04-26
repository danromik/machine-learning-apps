<script lang="ts">
  import Icon from './Icon.svelte';
  import { ui } from '../state.svelte';
  import { api } from '../api';

  let canvas: HTMLCanvasElement | undefined;
  let ctx: CanvasRenderingContext2D | null = null;
  let drawing = false;
  let last: { x: number; y: number } | null = null;

  // Offscreen 28x28 canvas powering the live "what the model sees" preview.
  let preview: HTMLCanvasElement | undefined;
  let previewCtx: CanvasRenderingContext2D | null = null;
  let previewRafPending = false;

  const SIZE = 168;

  function reset() {
    if (!ctx || !canvas) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }

  function updatePreview() {
    previewRafPending = false;
    if (!canvas || !preview || !previewCtx) return;
    previewCtx.clearRect(0, 0, 28, 28);
    previewCtx.drawImage(canvas, 0, 0, 28, 28);
    const url = preview.toDataURL('image/png');
    ui.predPreviewB64 = url.slice(url.indexOf(',') + 1);
  }

  function schedulePreview() {
    if (previewRafPending) return;
    previewRafPending = true;
    requestAnimationFrame(updatePreview);
  }

  function pos(e: MouseEvent | TouchEvent) {
    if (!canvas) return { x: 0, y: 0 };
    const r = canvas.getBoundingClientRect();
    const t = (e as TouchEvent).touches?.[0];
    const cx = t ? t.clientX : (e as MouseEvent).clientX;
    const cy = t ? t.clientY : (e as MouseEvent).clientY;
    return { x: ((cx - r.left) * canvas.width) / r.width, y: ((cy - r.top) * canvas.height) / r.height };
  }

  function down(e: MouseEvent | TouchEvent) {
    e.preventDefault();
    drawing = true;
    last = pos(e);
  }
  function move(e: MouseEvent | TouchEvent) {
    e.preventDefault();
    if (!drawing || !ctx || !last) return;
    const p = pos(e);
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last = p;
    schedulePreview();
  }
  function up(e: MouseEvent | TouchEvent) {
    e.preventDefault();
    drawing = false;
    schedulePreview();
  }

  $effect(() => {
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    reset();
    if (!preview) {
      preview = document.createElement('canvas');
      preview.width = 28;
      preview.height = 28;
    }
    previewCtx = preview.getContext('2d', { willReadFrequently: true });
    if (previewCtx) {
      previewCtx.imageSmoothingEnabled = true;
      previewCtx.imageSmoothingQuality = 'high';
    }
  });

  function clear() {
    reset();
    ui.classPred = null;
    ui.predProbs = Array(10).fill(0);
    ui.predPreviewB64 = '';
  }

  async function loadAndClassify(b64: string) {
    if (!canvas || !ctx) return;
    const img = new Image();
    img.src = `data:image/png;base64,${b64}`;
    try {
      await img.decode();
    } catch {
      return;
    }
    reset();
    const prevSmoothing = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = prevSmoothing;
    updatePreview();
    await classify();
  }

  // Loading a sample from Data Explorer: draw it and classify.
  $effect(() => {
    const req = ui.loadImage;
    if (!req || !ctx) return;
    void loadAndClassify(req.png_b64);
  });

  async function classify() {
    if (!canvas) return;
    const data_url = canvas.toDataURL('image/png');
    try {
      const res = await api.predict(data_url);
      ui.classPred = res.pred;
      ui.predProbs = res.probs;
      ui.predPreviewB64 = res.preview_b64;
      if (res.ckpt_name) ui.checkpointBadge = res.ckpt_name;
    } catch (e) {
      ui.status = `predict failed: ${(e as Error).message}`;
    }
  }
</script>

<section class="card flex flex-col min-h-0 w-80 shrink-0 overflow-auto">
  <div class="px-4 pt-3 pb-2">
    <h2 class="text-sm font-semibold tracking-tight">Classifier</h2>
  </div>
  <div class="px-4 pb-3 space-y-2">
    <div class="relative mx-auto" style="width: {SIZE}px">
      <canvas
        bind:this={canvas}
        width={SIZE}
        height={SIZE}
        class="block rounded-lg bg-black cursor-crosshair select-none"
        style="touch-action: none"
        onmousedown={down}
        onmousemove={move}
        onmouseup={up}
        onmouseleave={up}
        ontouchstart={down}
        ontouchmove={move}
        ontouchend={up}
      ></canvas>
      <span
        class="absolute top-1 left-1 px-1.5 py-0.5 text-white/70 text-[10px] leading-none pointer-events-none select-none"
      >
        Draw a digit to classify
      </span>
      <button
        class="absolute top-1 right-1 rounded px-1.5 py-0.5 bg-black/60 hover:bg-black text-white text-[10px] leading-none transition-colors"
        onclick={clear}
        aria-label="Clear canvas"
      >
        clear
      </button>
    </div>

    <button class="btn-primary w-full" onclick={classify}>
      <Icon name="brain" size={14} /> Classify
    </button>

    <div class="grid grid-cols-2 gap-x-4 gap-y-1 pt-1 justify-items-center">
      <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
        what the model sees
      </span>
      <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
        predicted value
      </span>
      <div
        class="w-14 h-14 rounded border border-[var(--color-border)] bg-black flex items-center justify-center overflow-hidden"
      >
        {#if ui.predPreviewB64}
          <img
            src="data:image/png;base64,{ui.predPreviewB64}"
            alt="model input"
            class="pixel-img w-full h-full"
          />
        {/if}
      </div>
      <div
        class="text-4xl font-bold w-12 text-center tabular-nums leading-none flex items-center justify-center h-14"
        class:text-[var(--color-accent)]={ui.classPred != null}
      >
        {ui.classPred ?? '–'}
      </div>
    </div>

    <div class="pt-1 space-y-0.5">
      {#each Array(10) as _, i}
        <div class="flex items-center gap-2 h-4 text-xs">
          <span
            class="w-3 text-right font-mono"
            class:text-[var(--color-accent)]={ui.classPred === i}
          >{i}</span>
          <div class="flex-1 h-1.5 bg-[var(--color-surface-2)] rounded overflow-hidden">
            <div
              class="h-full bg-[var(--color-accent)] transition-[width] duration-200 ease-out"
              style="width: {((ui.predProbs[i] ?? 0) * 100).toFixed(1)}%"
            ></div>
          </div>
          <span class="w-9 text-right font-mono tabular-nums text-[var(--color-muted)]">
            {(ui.predProbs[i] ?? 0).toFixed(2)}
          </span>
        </div>
      {/each}
    </div>
  </div>
</section>

<script lang="ts">
  import type { Frame } from '../api';

  // Renders one Snake frame on a canvas. `size` is the target pixel size of
  // the longer board edge; cells are square and centered.
  let { frame, size = 360 }: { frame: Frame | null; size?: number } = $props();

  let canvas: HTMLCanvasElement | null = $state(null);
  let host: HTMLDivElement | null = $state(null);

  function cssVar(name: string, fallback: string): string {
    if (!host) return fallback;
    const v = getComputedStyle(host).getPropertyValue(name).trim();
    return v || fallback;
  }

  $effect(() => {
    if (!canvas || !frame) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cols = frame.width;
    const rows = frame.height;
    const cell = Math.floor(size / Math.max(cols, rows));
    const w = cell * cols;
    const h = cell * rows;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const bg = cssVar('--color-bg', '#0b0f14');
    const grid = cssVar('--color-border', '#1f2937');
    const accent = cssVar('--color-accent', '#22c55e');
    const accentHover = cssVar('--color-accent-hover', '#16a34a');
    const danger = cssVar('--color-danger', '#ef4444');

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // Subtle grid.
    ctx.strokeStyle = grid;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    for (let x = 0; x <= cols; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cell + 0.5, 0);
      ctx.lineTo(x * cell + 0.5, h);
      ctx.stroke();
    }
    for (let y = 0; y <= rows; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cell + 0.5);
      ctx.lineTo(w, y * cell + 0.5);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    const pad = Math.max(1, Math.floor(cell * 0.08));
    const r = Math.max(2, Math.floor(cell * 0.2));

    function roundRect(px: number, py: number, s: number, fill: string) {
      ctx!.fillStyle = fill;
      ctx!.beginPath();
      ctx!.roundRect(px * cell + pad, py * cell + pad, s, s, r);
      ctx!.fill();
    }
    const body = cell - pad * 2;

    // Food.
    if (frame.food) {
      ctx.fillStyle = danger;
      const [fx, fy] = frame.food;
      ctx.beginPath();
      ctx.arc(fx * cell + cell / 2, fy * cell + cell / 2, cell * 0.32, 0, Math.PI * 2);
      ctx.fill();
    }

    // Snake — tail dimmer, head brightest.
    const snake = frame.snake;
    for (let i = snake.length - 1; i >= 0; i--) {
      const [sx, sy] = snake[i];
      if (i === 0) {
        roundRect(sx, sy, body, accentHover);
      } else {
        ctx.globalAlpha = 0.55 + 0.45 * (1 - i / Math.max(1, snake.length));
        roundRect(sx, sy, body, accent);
        ctx.globalAlpha = 1;
      }
    }
  });
</script>

<div bind:this={host} class="inline-block">
  {#if frame}
    <canvas bind:this={canvas} class="rounded-lg border border-[var(--color-border)]"></canvas>
  {:else}
    <div
      class="rounded-lg border border-dashed border-[var(--color-border)]
             flex items-center justify-center text-[var(--color-muted)] text-sm"
      style="width:{size}px;height:{size}px"
    >
      No game to show yet
    </div>
  {/if}
</div>

<script lang="ts">
  import { slide } from 'svelte/transition';
  import Icon from './Icon.svelte';
  import NetworkDiagram from './NetworkDiagram.svelte';
  import { cfg, ui, chartData } from '../state.svelte';
  import { api, type Architecture } from '../api';

  let editOpen = $state(false);
  let architecture = $state<Architecture | null>(null);

  // Local edit-buffer values used while the panel is open. Committed to cfg
  // when the user presses Save (or reverted on Cancel).
  let buf = $state({
    epochs: cfg.epochs,
    batch_size: cfg.batch_size,
    lr: cfg.lr,
    seed: cfg.seed,
  });

  $effect(() => {
    if (editOpen) {
      buf.epochs = cfg.epochs;
      buf.batch_size = cfg.batch_size;
      buf.lr = cfg.lr;
      buf.seed = cfg.seed;
    }
  });

  async function refreshParams() {
    try {
      const { params } = await api.params(cfg.model);
      ui.params = params;
    } catch (e) {
      ui.status = `params fetch failed: ${(e as Error).message}`;
      console.error('params fetch failed', e);
    }
  }

  async function refreshArchitecture() {
    try {
      architecture = await api.architecture(cfg.model);
    } catch (e) {
      console.error('architecture fetch failed', e);
      architecture = null;
    }
  }

  $effect(() => {
    cfg.model;
    refreshParams();
    refreshArchitecture();
  });

  function saveEdits() {
    cfg.epochs = Number(buf.epochs);
    cfg.batch_size = Number(buf.batch_size);
    cfg.lr = Number(buf.lr);
    cfg.seed = Number(buf.seed);
    editOpen = false;
  }
  function cancelEdits() {
    editOpen = false;
  }

  async function startSingle() {
    if (ui.training) return;
    await api.trainStart({ ...cfg, max_steps: 1 });
  }
  async function startSingleEpoch() {
    if (ui.training) return;
    await api.trainStart({ ...cfg, max_epochs: 1 });
  }
  async function startContinuous() {
    if (ui.isContinuous || ui.training) {
      await api.trainStop();
      ui.status = 'stop requested…';
      return;
    }
    ui.isContinuous = true;
    await api.trainStart({ ...cfg, max_steps: null });
  }
  async function reinit() {
    if (ui.training) return;
    try {
      await api.resetSession({ ...cfg });
      chartData.steps = [];
      chartData.losses = [];
      chartData.epochs = [];
      chartData.valLosses = [];
      chartData.valAccs = [];
      ui.cycles = 0;
      ui.checkpointBadge = '(none)';
      ui.classPred = null;
      ui.predProbs = Array(10).fill(0);
      ui.status = 'Network initialized with random weights. Ready to start training.';
    } catch (e) {
      ui.status = `reset failed: ${(e as Error).message}`;
    }
  }
  async function loadCheckpoint() {
    if (!ui.selectedCheckpoint) return;
    try {
      const res = await api.loadCheckpoint(ui.selectedCheckpoint, {
        epochs: cfg.epochs,
        batch_size: cfg.batch_size,
        lr: cfg.lr,
        seed: cfg.seed,
      });
      cfg.model = res.model as 'mlp' | 'cnn';
      ui.cycles = res.step;
      ui.checkpointBadge = ui.selectedCheckpoint;
      ui.status = `loaded · ${res.model} · ${res.step.toLocaleString()} cycles · best_acc ${res.best_acc.toFixed(4)}`;
    } catch (e) {
      ui.status = `load failed: ${(e as Error).message}`;
    }
  }
  async function saveCheckpoint() {
    if (ui.training) return;
    try {
      const { name } = await api.saveCheckpoint();
      const { files, current } = await api.checkpoints();
      ui.checkpoints = files;
      ui.selectedCheckpoint = name;
      if (current) ui.checkpointBadge = current;
    } catch (e) {
      ui.status = `save failed: ${(e as Error).message}`;
    }
  }
  async function onAutosaveChange() {
    await api.autosave(ui.autoSave);
  }

  function fmtLr(v: number) {
    return v.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }
</script>

<section class="card flex flex-col min-h-0 w-64 shrink-0 overflow-auto">
  <!-- Section title row, with edit pencil on the right -->
  <div class="flex items-center justify-between px-4 pt-3 pb-2">
    <h2 class="text-sm font-semibold tracking-tight">Neural Network</h2>
    <button
      class="btn-ghost p-1"
      title={editOpen ? 'Close' : 'Edit parameters & view architecture'}
      onclick={() => (editOpen = !editOpen)}
      aria-label="Edit parameters"
      aria-expanded={editOpen}
    >
      <Icon name={editOpen ? 'x' : 'edit'} size={14} />
    </button>
  </div>

  <!-- Drop-down editor + architecture panel -->
  {#if editOpen}
    <div
      transition:slide={{ duration: 180 }}
      class="px-4 pb-3 border-b border-[var(--color-border)] space-y-3"
    >
      <div class="grid grid-cols-2 gap-2">
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Epochs</span>
          <input type="number" class="input mt-0.5" min="1" max="50" bind:value={buf.epochs} />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Batch size</span>
          <input type="number" class="input mt-0.5" min="8" max="1024" bind:value={buf.batch_size} />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Learning rate</span>
          <input type="number" class="input mt-0.5" step="0.0001" bind:value={buf.lr} />
        </label>
        <label class="block">
          <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Seed</span>
          <input type="number" class="input mt-0.5" bind:value={buf.seed} />
        </label>
      </div>
      <div class="flex justify-end gap-1.5">
        <button class="btn-outline" onclick={cancelEdits}>Cancel</button>
        <button class="btn-primary" onclick={saveEdits}>
          <Icon name="check" size={14} /> Save
        </button>
      </div>

      <!-- Architecture visualization -->
      <div class="pt-1">
        <div class="flex items-center justify-between mb-1">
          <span class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
            Architecture
          </span>
          {#if architecture}
            <span class="text-[10px] text-[var(--color-muted)] font-mono">
              {architecture.layers.length} layers
            </span>
          {/if}
        </div>
        {#if architecture}
          <div class="rounded-md border border-[var(--color-border)] bg-[var(--color-surface)]/50">
            <NetworkDiagram layers={architecture.layers} />
          </div>
        {:else}
          <div class="text-xs text-[var(--color-muted)] py-2">loading…</div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Persistent body -->
  <div class="px-4 pb-4 pt-3 space-y-3">
    <label class="flex items-center gap-2">
      <span class="text-xs text-[var(--color-muted)] w-14">NN type</span>
      <select class="input flex-1" bind:value={cfg.model}>
        <option value="mlp">MLP</option>
        <option value="cnn">CNN</option>
      </select>
    </label>

    <div class="text-xs text-[var(--color-muted)]">
      Parameters
      <span class="font-mono text-[var(--color-text)] ml-1">
        {ui.params.toLocaleString()}
      </span>
    </div>

    <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
      <div><span class="text-[var(--color-muted)]">Epochs:</span> {cfg.epochs}</div>
      <div><span class="text-[var(--color-muted)]">Batch:</span> {cfg.batch_size}</div>
      <div><span class="text-[var(--color-muted)]">LR:</span> {fmtLr(cfg.lr)}</div>
      <div><span class="text-[var(--color-muted)]">Seed:</span> {cfg.seed}</div>
    </div>

    <div class="flex flex-col gap-1.5 pt-1">
      <button class="btn-outline w-full" onclick={reinit} disabled={ui.training}>
        <Icon name="restart" size={14} /> Re-Initialize
      </button>
      <button class="btn-outline w-full" onclick={startSingle} disabled={ui.training}>
        <Icon name="step" size={14} /> Train (Single Batch)
      </button>
      <button class="btn-outline w-full" onclick={startSingleEpoch} disabled={ui.training}>
        <Icon name="step" size={14} /> Train (Single Epoch)
      </button>
      {#if ui.isContinuous}
        <button class="btn-danger w-full" onclick={startContinuous}>
          <Icon name="stop" size={14} /> Stop Training
        </button>
      {:else}
        <button class="btn-primary w-full" onclick={startContinuous}>
          <Icon name="play" size={14} /> Train (Continuous)
        </button>
      {/if}
    </div>
  </div>

  <div class="px-4 py-3 border-t border-[var(--color-border)]">
    <h2 class="text-sm font-semibold tracking-tight mb-2">Checkpoints</h2>
    <label class="flex items-center gap-2">
      <span class="text-xs text-[var(--color-muted)] w-10">File</span>
      <select class="input flex-1" bind:value={ui.selectedCheckpoint}>
        {#if ui.checkpoints.length === 0}
          <option value="" disabled>(none yet)</option>
        {/if}
        {#each ui.checkpoints as f}
          <option value={f}>{f}</option>
        {/each}
      </select>
    </label>
    <div class="flex gap-1.5 mt-2">
      <button
        class="btn-outline flex-1"
        onclick={loadCheckpoint}
        disabled={!ui.selectedCheckpoint}
      >
        Load
      </button>
      <button class="btn-outline flex-1" onclick={saveCheckpoint} disabled={ui.training}>
        Save
      </button>
    </div>
    <label class="flex items-center gap-2 mt-3 cursor-pointer text-xs select-none">
      <input
        type="checkbox"
        class="accent-[var(--color-accent)]"
        bind:checked={ui.autoSave}
        onchange={onAutosaveChange}
      />
      <span>Automatically Save Checkpoint</span>
    </label>
  </div>
</section>

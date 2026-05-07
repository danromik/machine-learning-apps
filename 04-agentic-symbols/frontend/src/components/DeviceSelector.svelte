<script lang="ts">
  import Icon from './Icon.svelte';
  import { api, type DeviceList } from '../api';
  import { training } from '../state.svelte';

  let open = $state(false);
  let info = $state<DeviceList | null>(null);
  let busy = $state(false);
  let errorMsg = $state<string | null>(null);

  let buttonEl: HTMLButtonElement | undefined = $state();
  let dropdownEl: HTMLDivElement | undefined = $state();

  async function refresh() {
    try {
      info = await api.deviceList();
    } catch (e) {
      errorMsg = (e as Error).message;
    }
  }

  // Initial fetch.
  $effect(() => {
    refresh();
  });

  // Outside-click + ESC-to-close handler. Only attached while the
  // dropdown is open so we're not paying the listener cost otherwise.
  // Refreshes every time we open so the session-loaded indicator and
  // param count stay in sync with whatever happened in the Training tab.
  $effect(() => {
    if (!open) return;
    refresh();
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (
        (buttonEl && buttonEl.contains(t)) ||
        (dropdownEl && dropdownEl.contains(t))
      ) {
        return;
      }
      open = false;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') open = false;
    };
    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKey);
    };
  });

  async function selectDevice(name: string) {
    if (!info || busy) return;
    if (info.current === name) {
      open = false;
      return;
    }
    busy = true;
    errorMsg = null;
    try {
      await api.deviceSelect(name);
      await refresh();
      open = false;
    } catch (e) {
      errorMsg = (e as Error).message;
    } finally {
      busy = false;
    }
  }

  function formatMemory(bytes: number | null | undefined): string {
    if (!bytes) return '—';
    if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(1)} TB`;
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
    if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
    return `${bytes} B`;
  }

  function formatClock(hz: number | null | undefined): string | null {
    if (!hz) return null;
    if (hz >= 1e9) return `${(hz / 1e9).toFixed(2)} GHz`;
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(0)} MHz`;
    return `${hz} Hz`;
  }

  function formatParams(n: number): string {
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
    return n.toLocaleString();
  }
</script>

<div class="device-selector relative">
  <button
    bind:this={buttonEl}
    type="button"
    class="device-selector-button text-xs"
    class:device-selector-button-active={open}
    onclick={() => (open = !open)}
    aria-haspopup="true"
    aria-expanded={open}
    title="Choose training/inference device"
  >
    <Icon name="cpu" size={14} />
    <span class="text-[var(--color-muted)]">device:</span>
    <span class="font-mono text-[var(--color-text)]">
      {info?.current ?? '—'}
    </span>
  </button>

  {#if open && info}
    <div
      bind:this={dropdownEl}
      class="device-selector-dropdown card"
      role="dialog"
      aria-label="Device selector"
    >
      <header class="device-selector-section">
        <h4 class="text-xs font-semibold text-[var(--color-heading)]">
          Compute device
        </h4>
        {#if busy}
          <span class="text-[10px] text-[var(--color-muted)]">switching…</span>
        {/if}
      </header>

      <ul class="flex flex-col">
        {#each info.devices as d}
          {@const selected = d.name === info.current}
          {@const clock = formatClock(d.clock_hz)}
          <li>
            <button
              type="button"
              class="device-option"
              class:device-option-selected={selected}
              disabled={busy || !d.available}
              onclick={() => selectDevice(d.name)}
            >
              <div class="flex items-center gap-2 w-full">
                <span
                  class="device-option-name font-mono uppercase tracking-wide"
                >{d.name}</span>
                <span class="font-medium truncate flex-1 text-left">
                  {d.label}
                </span>
                {#if selected}
                  <Icon
                    name="check"
                    size={14}
                    class="text-[var(--color-accent)] shrink-0"
                  />
                {/if}
              </div>
              <div
                class="device-option-meta text-[10px] text-[var(--color-muted)] font-mono"
              >
                {#if d.cores !== undefined}{d.cores} cores · {/if}
                {#if clock}{clock} · {/if}
                {formatMemory(d.memory_bytes)} memory
                {#if d.memory_note} ({d.memory_note}){/if}
              </div>
            </button>
          </li>
        {/each}
      </ul>

      <footer class="device-selector-section device-selector-footer">
        <div class="flex items-center justify-between gap-2 text-[11px]">
          <span class="text-[var(--color-muted)]">model</span>
          <!-- Read from training state directly (not info.session_loaded
               which is only refreshed when the dropdown opens) so the
               indicator flips off immediately when a Data Synthesis
               change resets the session. -->
          {#if training.hasSession}
            <span
              class="font-mono text-[var(--color-success)] flex items-center gap-1"
            >
              <Icon name="check" size={12} /> loaded ·
              {formatParams(training.paramCount)} params
            </span>
          {:else}
            <span class="font-mono text-[var(--color-muted)]">
              not loaded
            </span>
          {/if}
        </div>
      </footer>

      {#if errorMsg}
        <div class="px-3 pb-2 text-[10px] text-[var(--color-danger)]">
          {errorMsg}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* Persistent button affordance — subtle pill shape with a border so
     the control reads as clickable without needing a disclosure chevron.
     Hover/active states layer on a stronger surface tint. */
  .device-selector-button {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.3rem 0.625rem;
    border-radius: 9999px;
    border: 1px solid var(--color-border);
    background: var(--color-surface-2);
    color: var(--color-text);
    cursor: pointer;
    user-select: none;
    transition: background-color 100ms, border-color 100ms, filter 100ms;
  }
  .device-selector-button:hover {
    background: var(--color-surface);
    border-color: var(--color-accent);
    filter: brightness(1.04);
  }
  .device-selector-button-active {
    background: var(--color-surface);
    border-color: var(--color-accent);
  }
  .device-selector-dropdown {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 30;
    width: 18rem;
    box-shadow:
      0 8px 24px rgba(0, 0, 0, 0.18),
      0 2px 6px rgba(0, 0, 0, 0.08);
    overflow: hidden;
  }
  .device-selector-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--color-border);
  }
  .device-selector-footer {
    border-bottom: none;
    border-top: 1px solid var(--color-border);
    background: var(--color-surface-2);
  }
  .device-option {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.125rem;
    width: 100%;
    padding: 0.5rem 0.75rem;
    text-align: left;
    background: transparent;
    border: none;
    cursor: pointer;
    color: var(--color-text);
    transition: background-color 100ms;
  }
  .device-option:hover:not(:disabled) {
    background: var(--color-surface-2);
  }
  .device-option:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .device-option-selected {
    background: color-mix(in oklab, var(--color-accent) 8%, transparent);
  }
  .device-option-selected:hover:not(:disabled) {
    background: color-mix(in oklab, var(--color-accent) 14%, transparent);
  }
  .device-option-name {
    font-size: 9px;
    color: var(--color-muted);
    background: var(--color-surface-2);
    border: 1px solid var(--color-border);
    padding: 0 0.25rem;
    border-radius: 0.2rem;
    line-height: 1.4;
  }
  .device-option-meta {
    padding-left: calc(0.25rem + 1.5rem);
    /* Indented under the name so the "12 cores · 32 GB memory" line
       reads as a property of the option above it, not its own item. */
  }
</style>

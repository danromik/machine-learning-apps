<!--
  Header row inside the chat pane: shows the active session's summary +
  a button that opens a dropdown of past sessions. Clicking a past
  session triggers the resume flow (handled in ChatPane). Plus button
  starts a fresh chat.
-->
<script lang="ts">
  import Icon from '../Icon.svelte';
  import { chat } from '../../state.svelte';
  import { api } from '../../api';

  let { onResume, onNewChat }: {
    onResume: (sessionId: string) => void;
    onNewChat: () => void;
  } = $props();

  let open = $state(false);
  let loading = $state(false);
  let menuRef: HTMLDivElement | null = $state(null);

  let activeSummary = $derived.by(() => {
    if (!chat.activeSessionId) return 'New chat';
    const found = chat.sessions.find(
      (s) => s.session_id === chat.activeSessionId,
    );
    return found?.summary || 'Conversation in progress';
  });

  async function refreshSessions() {
    loading = true;
    try {
      const res = await api.listChatSessions(50);
      chat.sessions = res.sessions;
    } catch (e) {
      console.warn('listChatSessions failed', e);
    } finally {
      loading = false;
    }
  }

  async function toggle() {
    if (!open) await refreshSessions();
    open = !open;
  }

  function pick(sessionId: string) {
    open = false;
    onResume(sessionId);
  }

  function newChat() {
    open = false;
    onNewChat();
  }

  // Close dropdown on outside click.
  function onWindowClick(e: MouseEvent) {
    if (!open) return;
    if (menuRef && !menuRef.contains(e.target as Node)) open = false;
  }

  function fmtRelative(ms: number | null): string {
    if (!ms) return '';
    const diff = Date.now() - ms;
    const min = Math.floor(diff / 60_000);
    if (min < 1) return 'just now';
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const day = Math.floor(hr / 24);
    if (day < 7) return `${day}d ago`;
    return new Date(ms).toLocaleDateString();
  }
</script>

<svelte:window onclick={onWindowClick} />

<div
  bind:this={menuRef}
  class="relative flex items-center justify-between px-3 py-2 border-b
         border-[var(--color-border)] shrink-0"
>
  <button
    type="button"
    onclick={toggle}
    class="flex items-center gap-1.5 text-xs text-[var(--color-text)] min-w-0
           hover:text-[var(--color-heading)] transition-colors"
  >
    <Icon name="message-square" size={13} class="shrink-0 text-[var(--color-muted)]" />
    <span class="truncate max-w-[180px]">{activeSummary}</span>
    <Icon name="chevron-down" size={12} class="shrink-0 text-[var(--color-muted)]" />
  </button>

  <button
    type="button"
    onclick={newChat}
    title="New chat"
    class="text-[var(--color-muted)] hover:text-[var(--color-heading)] transition-colors"
  >
    <Icon name="square-pen" size={14} />
  </button>

  {#if open}
    <div
      class="absolute left-2 top-full mt-1 w-[280px] max-h-[360px] overflow-auto
             rounded-md border border-[var(--color-border)] bg-[var(--color-surface)]
             shadow-xl z-30 text-xs"
    >
      <div
        class="px-3 py-2 text-[10px] uppercase tracking-wider text-[var(--color-muted)]
               border-b border-[var(--color-border)]/60 sticky top-0 bg-[var(--color-surface)]"
      >
        {loading ? 'loading…' : `${chat.sessions.length} sessions`}
      </div>
      {#if chat.sessions.length === 0 && !loading}
        <div class="px-3 py-4 text-[var(--color-muted)] italic">
          No past sessions yet.
        </div>
      {/if}
      {#each chat.sessions as s (s.session_id)}
        <button
          type="button"
          onclick={() => pick(s.session_id)}
          class="w-full text-left px-3 py-2 hover:bg-[var(--color-border)]/40
                 transition-colors flex flex-col gap-0.5"
          class:bg-[var(--color-border)]={s.session_id === chat.activeSessionId}
          class:bg-opacity-50={s.session_id === chat.activeSessionId}
        >
          <span class="truncate text-[var(--color-text)]">
            {s.summary || '(untitled)'}
          </span>
          <span class="text-[10px] text-[var(--color-muted)]">
            {fmtRelative(s.last_modified)}
          </span>
        </button>
      {/each}
    </div>
  {/if}
</div>

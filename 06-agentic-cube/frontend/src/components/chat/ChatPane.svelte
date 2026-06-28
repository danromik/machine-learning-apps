<!--
  RL Coach chat pane. Composes the usage bar, session menu,
  transcript, and input — and owns the SSE streaming loop that turns
  agent events into ChatItems.

  Streaming reducer:
    text_delta   → append to last assistant text item (or open one)
    text_message → close current bubble (emitted at content-block boundary)
    tool_use     → close any open bubble, push a tool item with status=running
    tool_result  → find tool by id, set result + status=success/error
    usage        → update token counts (drives UsageBar)
    result       → finalize: set sessionId, persist usage, close turn
    error        → push an error item, close turn
-->
<script lang="ts">
  import {
    chat,
    clampChatPaneWidth,
    persistChatPrefs,
    type ChatItem,
    type ChatSessionSummary,
  } from '../../state.svelte';
  import { api, streamChat, type ChatEvent } from '../../api';
  import { applyChatEvent } from './chatReducer';
  import UsageBar from './UsageBar.svelte';
  import SessionMenu from './SessionMenu.svelte';
  import ChatTranscript from './ChatTranscript.svelte';
  import Icon from '../Icon.svelte';

  let abortController: AbortController | null = null;
  let textareaRef: HTMLTextAreaElement | null = $state(null);

  // The transcript reducer is shared with the WS-bridged autonomous check-in
  // path (see chatReducer.ts + App.svelte's agent_event handler).
  const applyEvent = applyChatEvent;

  // ── Send / stop ────────────────────────────────────────────────────

  async function send() {
    const text = chat.draft.trim();
    if (!text || chat.turn === 'streaming') return;

    chat.items.push({ kind: 'text', role: 'user', text, streaming: false });
    chat.draft = '';
    chat.turn = 'streaming';

    abortController = new AbortController();
    try {
      for await (const ev of streamChat(
        text,
        chat.activeSessionId,
        abortController.signal,
      )) {
        applyEvent(ev);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      // Abort throws DOMException with name=AbortError — we surface it
      // as a tagged error item so the user sees something happened.
      if (!/abort/i.test(msg)) {
        chat.items.push({ kind: 'error', message: msg });
      }
    } finally {
      chat.turn = 'idle';
      abortController = null;
    }
  }

  function stop() {
    if (abortController) abortController.abort();
    api.stopChat().catch(() => {
      // Best-effort — the backend may have already finished.
    });
  }

  // ── Resume / new ──────────────────────────────────────────────────

  async function resume(sessionId: string) {
    if (chat.turn === 'streaming') stop();
    chat.activeSessionId = sessionId;
    chat.items = [];
    persistChatPrefs();
    try {
      const { events } = await api.loadChatSession(sessionId);
      for (const ev of events) applyEvent(ev as ChatEvent);
    } catch (e) {
      console.warn('loadChatSession failed', e);
      chat.items.push({
        kind: 'error',
        message: `Couldn't load session: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }

  function newChat() {
    if (chat.turn === 'streaming') stop();
    chat.activeSessionId = null;
    chat.items = [];
    chat.usage = {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
      total_cost_usd: 0,
    };
    persistChatPrefs();
    queueMicrotask(() => textareaRef?.focus());
  }

  // ── Composer keybindings ──────────────────────────────────────────

  function onKeydown(e: KeyboardEvent) {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  // Auto-resize the textarea to fit content (up to a max).
  function onInput() {
    if (!textareaRef) return;
    textareaRef.style.height = 'auto';
    textareaRef.style.height = Math.min(textareaRef.scrollHeight, 220) + 'px';
  }

  // ── Divider drag-to-resize ────────────────────────────────────────
  //
  // Mousedown on the divider captures the starting cursor x and pane
  // width; mousemove updates `chat.paneWidth` live (Svelte reflows the
  // aside's `style="width:"` and the parent flex shrinks `<main>` to
  // match). Mouseup persists the final width and tears down listeners.
  // Listeners hang off `window` so a drag past the chrome still
  // releases cleanly. Body cursor + user-select are pinned across the
  // drag so nothing under the cursor flickers or selects text.

  let dragging = $state(false);

  function onDividerPointerDown(e: PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    dragging = true;
    const startX = e.clientX;
    const startWidth = chat.paneWidth;
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    function onMove(ev: PointerEvent) {
      // Pane is right-anchored — dragging the divider left grows it,
      // dragging right shrinks it. So subtract dx from startWidth.
      chat.paneWidth = clampChatPaneWidth(startWidth - (ev.clientX - startX));
    }
    function onUp() {
      dragging = false;
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      persistChatPrefs();
    }
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  }
</script>

<!--
  Divider / resize handle. Sits as a sibling of the aside in App.svelte's
  parent flex row, so dragging it shrinks <main> and grows <aside> in
  one reflow. Visually 4px wide; tinted with the accent on hover or
  while dragging so the affordance is obvious.
-->
<div
  role="separator"
  aria-orientation="vertical"
  aria-label="Resize chat pane"
  class="w-1 shrink-0 cursor-col-resize transition-colors
         {dragging
           ? 'bg-[var(--color-accent)]'
           : 'bg-[var(--color-border)] hover:bg-[var(--color-accent)]'}"
  onpointerdown={onDividerPointerDown}
></div>

<aside
  class="h-full shrink-0 flex flex-col bg-[var(--color-bg)] min-h-0"
  style="width: {chat.paneWidth}px"
>
  <UsageBar />
  <SessionMenu onResume={resume} onNewChat={newChat} />
  <ChatTranscript />

  <form
    onsubmit={(e) => { e.preventDefault(); send(); }}
    class="border-t border-[var(--color-border)] p-2.5 flex items-end gap-2 shrink-0"
  >
    <textarea
      bind:this={textareaRef}
      bind:value={chat.draft}
      oninput={onInput}
      onkeydown={onKeydown}
      placeholder="Ask the RL Coach…"
      rows="1"
      class="flex-1 resize-none rounded-md border border-[var(--color-border)]
             bg-[var(--color-surface)] px-2.5 py-1.5 text-sm leading-snug
             text-[var(--color-text)] placeholder:text-[var(--color-muted)]
             focus:outline-none focus:border-[var(--color-accent)]
             max-h-[220px] min-h-[34px]"
    ></textarea>
    {#if chat.turn === 'streaming'}
      <button
        type="button"
        onclick={stop}
        class="shrink-0 h-[34px] px-3 rounded-md bg-rose-500 text-white text-sm
               font-medium hover:bg-rose-600 transition-colors flex items-center gap-1.5"
      >
        <Icon name="stop" size={13} />
        Stop
      </button>
    {:else}
      <button
        type="submit"
        disabled={!chat.draft.trim()}
        class="shrink-0 h-[34px] px-3 rounded-md bg-[var(--color-accent)] text-white
               text-sm font-medium hover:opacity-90 transition-opacity disabled:opacity-40
               disabled:cursor-not-allowed flex items-center gap-1.5"
      >
        <Icon name="send" size={13} />
        Send
      </button>
    {/if}
  </form>
</aside>

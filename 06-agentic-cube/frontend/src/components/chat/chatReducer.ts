// The single reducer that folds a ChatEvent into the `chat` store's item
// list. Used by BOTH the live SSE stream (user-initiated turns, in ChatPane)
// AND the /ws/state-bridged `agent_event` stream (autonomous RL Coach
// check-ins, dispatched from App.svelte) so the transcript renders identically
// no matter who drove the turn.

import { chat, persistChatPrefs, type ChatItem } from '../../state.svelte';
import type { ChatEvent } from '../../api';

function closeOpenBubble() {
  const last = chat.items[chat.items.length - 1];
  if (last && last.kind === 'text' && last.streaming) last.streaming = false;
}

function findToolById(id: string): Extract<ChatItem, { kind: 'tool' }> | null {
  for (let i = chat.items.length - 1; i >= 0; i--) {
    const it = chat.items[i];
    if (it.kind === 'tool' && it.id === id) return it;
  }
  return null;
}

export function applyChatEvent(ev: ChatEvent) {
  if (ev.type === 'text_delta') {
    const last = chat.items[chat.items.length - 1];
    if (last && last.kind === 'text' && last.role === 'assistant' && last.streaming) {
      last.text += ev.text;
    } else {
      chat.items.push({ kind: 'text', role: 'assistant', text: ev.text, streaming: true });
    }
  } else if (ev.type === 'text_message') {
    const last = chat.items[chat.items.length - 1];
    if (last && last.kind === 'text' && last.role === 'assistant' && last.streaming) {
      if (ev.text && ev.text.length > 0) last.text = ev.text;
      last.streaming = false;
    } else if (ev.text && ev.text.length > 0) {
      chat.items.push({ kind: 'text', role: 'assistant', text: ev.text, streaming: false });
    }
  } else if (ev.type === 'tool_use') {
    closeOpenBubble();
    chat.items.push({ kind: 'tool', id: ev.id, name: ev.name, input: ev.input, status: 'running' });
  } else if (ev.type === 'tool_result') {
    const tool = findToolById(ev.tool_use_id);
    if (tool) {
      tool.result = ev.content;
      tool.status = ev.is_error ? 'error' : 'success';
    }
  } else if (ev.type === 'user_message') {
    chat.items.push({ kind: 'text', role: 'user', text: ev.text, streaming: false });
  } else if (ev.type === 'checkin_start') {
    // Autonomous check-in separator so the transcript shows the Coach woke
    // itself up (rather than the user sending a message).
    closeOpenBubble();
    chat.items.push({ kind: 'system', text: `RL Coach check-in — ${ev.reason}` });
  } else if (ev.type === 'usage') {
    if (typeof ev.input_tokens === 'number') chat.usage.input_tokens = ev.input_tokens;
    if (typeof ev.output_tokens === 'number') chat.usage.output_tokens = ev.output_tokens;
    if (typeof ev.cache_read_input_tokens === 'number')
      chat.usage.cache_read_input_tokens = ev.cache_read_input_tokens;
    if (typeof ev.cache_creation_input_tokens === 'number')
      chat.usage.cache_creation_input_tokens = ev.cache_creation_input_tokens;
  } else if (ev.type === 'result') {
    closeOpenBubble();
    if (ev.session_id && ev.session_id !== chat.activeSessionId) {
      chat.activeSessionId = ev.session_id;
      persistChatPrefs();
    }
    if (typeof ev.total_cost_usd === 'number') chat.usage.total_cost_usd += ev.total_cost_usd;
    const u = ev.usage as Record<string, number> | null;
    if (u) {
      if (typeof u.input_tokens === 'number') chat.usage.input_tokens = u.input_tokens;
      if (typeof u.output_tokens === 'number') chat.usage.output_tokens = u.output_tokens;
    }
  } else if (ev.type === 'error') {
    closeOpenBubble();
    chat.items.push({ kind: 'error', message: ev.message });
  }
}

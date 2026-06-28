async function j<T>(p: Promise<Response>): Promise<T> {
  const res = await p;
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Domain types ────────────────────────────────────────────────────────

export type AlgorithmId = 'qlearning' | 'dqn' | 'reinforce';

export type AlgorithmInfo = {
  id: AlgorithmId;
  label: string;
  uses_network: boolean;
  description: string;
  default_hyperparameters: Record<string, number>;
};

export type RewardConfig = {
  food: number;
  death: number;
  step: number;
  toward_food: number;
  away_from_food: number;
};

export type ObservationModel = 'features' | 'grid';

export type EnvironmentConfig = {
  width: number;
  height: number;
  observation: ObservationModel;
  reward: RewardConfig;
};

// One per-episode training record (also the shape of episode_tick.record).
export type EpisodeRecord = {
  episode: number;
  score: number;
  reward: number;
  length: number;
  epsilon?: number;
  loss?: number;
  td_error?: number;
  q_states?: number;
  buffer?: number;
};

export type SessionSummary = {
  has_session: boolean;
  algo?: AlgorithmId;
  uses_network?: boolean;
  param_count?: number;
  episode?: number;
  best_score?: number;
  device?: string;
};

export type SessionState = SessionSummary & { score_history?: EpisodeRecord[] };

// A single rendered game frame.
export type Frame = {
  width: number;
  height: number;
  snake: [number, number][];
  food: [number, number] | null;
  score: number;
  steps: number;
  done: boolean;
};

export type PlayStep = {
  action: number;
  reward: number;
  scores: { kind: 'q' | 'prob'; values: number[] } | null;
  event: string | null;
};

export type PlayResult = {
  frames: Frame[];
  steps: PlayStep[];
  score: number;
  length: number;
};

export type CheckpointFile = { name: string; size: number; mtime: number };

export type Device = {
  name: string;
  label: string;
  cores?: number;
  clock_hz?: number;
  memory_bytes?: number | null;
  memory_note?: string;
  available: boolean;
};

export type DeviceList = {
  current: string;
  devices: Device[];
  session_loaded: boolean;
  param_count: number;
};

export const ACTION_NAMES = ['straight', 'right', 'left'];

export const api = {
  device: () => j<{ device: string }>(fetch('/api/device')),
  deviceList: () => j<DeviceList>(fetch('/api/device/list')),
  deviceSelect: (name: string) =>
    j<{ current: string }>(
      fetch('/api/device/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })
    ),

  catalog: () => j<{ algorithms: AlgorithmInfo[] }>(fetch('/api/catalog')),

  // Training — session is built from the pipeline-state mirror.
  initTraining: () =>
    j<SessionSummary>(fetch('/api/training/init', { method: 'POST' })),
  trainingState: () => j<SessionState>(fetch('/api/training/state')),
  resetTraining: () =>
    j<{ has_session: false }>(fetch('/api/training/reset', { method: 'POST' })),
  trainEpisodes: (n: number, hyperparameters?: Record<string, number>) =>
    j<{ records: EpisodeRecord[]; summary: Record<string, number> }>(
      fetch('/api/training/train_episodes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n, hyperparameters }),
      })
    ),
  evaluate: (n: number) =>
    j<{ episodes: number; mean_score: number; best_score: number; mean_length: number }>(
      fetch('/api/training/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n }),
      })
    ),
  play: (greedy = true) =>
    j<PlayResult>(
      fetch('/api/training/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ greedy }),
      })
    ),

  listCheckpoints: () =>
    j<{ files: CheckpointFile[] }>(fetch('/api/training/checkpoints')),
  saveCheckpoint: (filename: string) =>
    j<{ name: string }>(
      fetch('/api/training/checkpoints/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
    ),
  loadCheckpoint: (filename: string) =>
    j<SessionState>(
      fetch('/api/training/checkpoints/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
    ),
  deleteCheckpoint: (filename: string) =>
    j<{ name: string }>(
      fetch('/api/training/checkpoints/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
    ),

  // Pipeline-state mirror
  getState: () => j<PipelineStateSnapshot>(fetch('/api/state')),
  patchState: (patch: Record<string, unknown>) =>
    j<{ ok: true }>(
      fetch('/api/state/patch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch }),
      })
    ),

  // RL Coach chat
  listChatSessions: (limit = 50) =>
    j<{ sessions: ChatSessionInfo[] }>(fetch(`/api/agent/sessions?limit=${limit}`)),
  loadChatSession: (sessionId: string) =>
    j<{ session_id: string; events: ChatEvent[] }>(
      fetch(`/api/agent/sessions/${encodeURIComponent(sessionId)}/messages`)
    ),
  stopChat: () =>
    j<{ cancelled: boolean; reason?: string }>(
      fetch('/api/agent/stop', { method: 'POST' })
    ),
};

// ── Pipeline state ──────────────────────────────────────────────────────

export type PipelineStateSnapshot = {
  environment: EnvironmentConfig;
  algorithm: { algo: AlgorithmId; hyperparameters: Record<string, number> };
  training: { episodesPerRun: number; evalEveryN: number };
};

// ── Chat types (ported from 04, domain-agnostic) ────────────────────────

export type ChatSessionInfo = {
  session_id: string;
  summary: string | null;
  message_count: number | null;
  last_modified: number | null;
  created_at: number | null;
};

export type ChatEvent =
  | { type: 'text_delta'; text: string }
  | { type: 'text_message'; text: string }
  | { type: 'user_message'; text: string }
  | { type: 'tool_use'; id: string; name: string; full_name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error: boolean }
  | {
      type: 'usage';
      input_tokens?: number;
      output_tokens?: number;
      cache_read_input_tokens?: number;
      cache_creation_input_tokens?: number;
    }
  | {
      type: 'result';
      session_id: string | null;
      total_cost_usd: number | null;
      duration_ms: number | null;
      is_error: boolean;
      subtype: string | null;
      usage: Record<string, unknown> | null;
    }
  | { type: 'error'; message: string };

// State broadcast events over /ws/state. The deep-merge mirror
// (state_replace / state_patch) plus transient training-progress signals
// (training_session / episode_tick) that flow into the `training` store.
export type StateBroadcastEvent =
  | { type: 'state_replace'; source: string; state: PipelineStateSnapshot }
  | { type: 'state_patch'; source: string; patch: Record<string, unknown> }
  | {
      type: 'training_session';
      source: string;
      hasSession: boolean;
      summary?: SessionSummary;
      scoreHistory?: EpisodeRecord[];
    }
  | { type: 'episode_tick'; source: string; record: EpisodeRecord };

/** Stream agent events from /api/agent/chat as parsed objects. */
export async function* streamChat(
  message: string,
  sessionId: string | null,
  signal?: AbortSignal
): AsyncGenerator<ChatEvent> {
  const response = await fetch('/api/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(detail || `HTTP ${response.status}`);
  }
  if (!response.body) throw new Error('chat stream has no body');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLines = frame
          .split('\n')
          .filter((l) => l.startsWith('data:'))
          .map((l) => l.slice(5).trimStart());
        if (dataLines.length === 0) continue;
        try {
          yield JSON.parse(dataLines.join('\n')) as ChatEvent;
        } catch {
          // Malformed frame — skip.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Open a WebSocket to /ws/state and forward parsed events. */
export function openStateSocket(
  onEvent: (ev: StateBroadcastEvent) => void,
  onClose?: (e: CloseEvent) => void
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${window.location.host}/ws/state`;
  const ws = new WebSocket(url);
  ws.addEventListener('message', (ev) => {
    try {
      onEvent(JSON.parse(ev.data) as StateBroadcastEvent);
    } catch {
      // Malformed message — skip.
    }
  });
  if (onClose) ws.addEventListener('close', onClose);
  return ws;
}

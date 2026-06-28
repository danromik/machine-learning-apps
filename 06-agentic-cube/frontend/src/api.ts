async function j<T>(p: Promise<Response>): Promise<T> {
  const res = await p;
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Domain types ────────────────────────────────────────────────────────

export type AlgorithmId = 'value_iteration';

export type AlgorithmInfo = {
  id: AlgorithmId;
  label: string;
  uses_network: boolean;
  description: string;
  default_hyperparameters: Record<string, number>;
};

export type Curriculum = { startK: number; maxK: number; promoteAt: number };

export type EnvironmentConfig = {
  size: number; // 2 or 3
  curriculum: Curriculum;
};

// One per-iteration training record (also the shape of trainer_progress.record).
export type IterationRecord = {
  iteration: number;
  k: number;
  loss: number;
  mean_target: number;
};

export type SessionSummary = {
  has_session: boolean;
  algo?: AlgorithmId;
  cube_size?: number;
  uses_network?: boolean;
  param_count?: number;
  iteration?: number;
  current_k?: number;
  solve_rate_by_k?: Record<string, number>;
  device?: string;
};

export type SessionState = SessionSummary & { loss_history?: IterationRecord[] };

// ── 3D Watch view types ──────────────────────────────────────────────────
export type Sticker = { normal: [number, number, number]; color: number };
export type Cubie = { pos: [number, number, number]; stickers: Sticker[] };
export type CubeFrame = {
  size: number;
  coords: number[];
  cubies: Cubie[];
  solved: boolean;
};
export type MoveMeta = { name: string; axis: number; sign: number; dir: number };
export type SolveStep = {
  move: MoveMeta;
  scores: { kind: 'cost'; values: number[] } | null;
};
export type SolveResult = {
  size: number;
  scramble_depth: number;
  scramble_moves: string[];
  frames: CubeFrame[];
  steps: SolveStep[];
  solved: boolean;
  solution_len: number;
  move_catalog: MoveMeta[];
};

export type EvalResult = {
  attempted: number;
  k: number;
  solve_rate: number;
  mean_solution_len: number | null;
};

export type RunStatus = {
  state: string;
  running: boolean;
  run_id: string | null;
  error: string | null;
  started_at: number | null;
  iteration: number;
  current_k: number;
  solve_rate_by_k: Record<string, number>;
  last_checkpoint: string | null;
  last_checkpoint_at: number | null;
  config: Record<string, unknown> | null;
};

export type ReportState = {
  markdown: string;
  final: boolean;
  updated_at: number | null;
  run_id: string | null;
};

export type CheckpointFile = { name: string; size: number; mtime: number; protected?: boolean };

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
  initTraining: () => j<SessionSummary>(fetch('/api/training/init', { method: 'POST' })),
  trainingState: () => j<SessionState>(fetch('/api/training/state')),
  resetTraining: () =>
    j<{ has_session: false }>(fetch('/api/training/reset', { method: 'POST' })),
  trainIterations: (n: number, k?: number | null, hyperparameters?: Record<string, number>) =>
    j<{ records: IterationRecord[]; summary: Record<string, number> }>(
      fetch('/api/training/train_iterations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n, k, hyperparameters }),
      })
    ),
  evaluate: (n: number, k?: number | null) =>
    j<EvalResult>(
      fetch('/api/training/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n, k }),
      })
    ),
  play: (k?: number | null) =>
    j<SolveResult>(
      fetch('/api/training/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k }),
      })
    ),

  // Background run
  runStart: (cfg: Record<string, unknown>) =>
    j<RunStatus>(
      fetch('/api/training/run/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      })
    ),
  runStop: () => j<RunStatus>(fetch('/api/training/run/stop', { method: 'POST' })),
  runStatus: () => j<RunStatus>(fetch('/api/training/run/status')),
  runResume: () => j<RunStatus>(fetch('/api/training/run/resume', { method: 'POST' })),

  report: () => j<ReportState>(fetch('/api/training/report')),

  listCheckpoints: () => j<{ files: CheckpointFile[] }>(fetch('/api/training/checkpoints')),
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
    j<{ cancelled: boolean; reason?: string }>(fetch('/api/agent/stop', { method: 'POST' })),
};

// ── Pipeline state ──────────────────────────────────────────────────────

export type PipelineStateSnapshot = {
  environment: EnvironmentConfig;
  algorithm: { algo: AlgorithmId; hyperparameters: Record<string, number> };
  training: {
    iterationsPerRun: number;
    evalEveryN: number;
    evalN: number;
    cadenceMinutes: number;
  };
};

// ── Chat types (domain-agnostic) ────────────────────────────────────────

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
  | { type: 'checkin_start'; reason: string; run_id: string | null }
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

// State broadcast events over /ws/state: the deep-merge mirror
// (state_replace / state_patch) plus transient progress signals.
export type StateBroadcastEvent =
  | { type: 'state_replace'; source: string; state: PipelineStateSnapshot }
  | { type: 'state_patch'; source: string; patch: Record<string, unknown> }
  | {
      type: 'training_session';
      source: string;
      hasSession: boolean;
      summary?: SessionSummary;
      lossHistory?: IterationRecord[];
    }
  | {
      type: 'trainer_progress';
      source: string;
      record: IterationRecord;
      iteration: number;
      current_k: number;
      solve_rate_by_k: Record<string, number>;
    }
  | {
      type: 'trainer_status';
      source: string;
      state: string;
      run_id: string | null;
      error?: string | null;
      eval?: EvalResult;
      current_k?: number;
      promoted?: boolean;
      checkpoint?: string;
      iteration?: number;
    }
  | { type: 'report_update'; source: string; markdown: string; updated_at: number }
  | { type: 'report_final'; source: string; markdown: string; updated_at: number }
  | { type: 'agent_event'; source: string; event: ChatEvent };

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

async function j<T>(p: Promise<Response>): Promise<T> {
  const res = await p;
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export type SymbolCategory = {
  id: string;
  label: string;
  symbols: string[];
  count: number;
};

export type Font = {
  family: string;
  url: string;
  install: string;
  note: string;
  installed: boolean;
};

export type PresetName = 'beginner' | 'intermediate' | 'advanced';

export type SynthesisPreset = {
  name: PresetName;
  categories: string[];
  training_fonts: string[];
  validation_fonts: string[];
  augmentation: {
    noise: { enabled: boolean; max_level: number };
    skew: { enabled: boolean };
  };
};

// ── Training API types ─────────────────────────────────────────────────

export type TrainingHyperparameters = {
  lr: number;
  batch_size: number;
  optimizer: string;
};

export type SessionState =
  | { has_session: false }
  | {
      has_session: true;
      num_classes: number;
      param_count: number;
      step: number;
      lr: number;
      batch_size: number;
      optimizer: string;
    };

// Snapshot of the data synthesis pipeline saved alongside model weights.
// Mirrors the three fields the frontend tracks reactively (and uses to
// invalidate the session on change).
export type CheckpointSynthesisConfig = {
  selectedCategories: Record<string, boolean>;
  fontUsage: Record<string, 'off' | 'train' | 'val'>;
  augmentation: {
    noise: { enabled: boolean; max_level: number };
    skew: { enabled: boolean };
  };
};

// Layer payload as it lives in checkpoints — same shape `train_init` /
// `train_batch` use, minus the frontend-only `id` field.
export type CheckpointLayer = {
  type: string;
  params: Record<string, number>;
};

export type LossPoint = { step: number; loss: number };

export type CheckpointFile = {
  name: string;
  // Size in bytes; mtime in unix seconds. Frontend formats both for
  // display (e.g. "1.4 MB", "14:32 today").
  size: number;
  mtime: number;
};

export type CheckpointLoadResponse = SessionState & {
  has_session: true;
  layers: CheckpointLayer[];
  classes: string[];
  synthesis_config: CheckpointSynthesisConfig | null;
  loss_history: LossPoint[];
  val_loss_history: LossPoint[];
};

export type TrainBatchResult = { loss: number; step: number; accuracy: number };

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
  symbols: () =>
    j<{ categories: SymbolCategory[] }>(fetch('/api/synthesis/symbols')),
  fonts: () => j<{ fonts: Font[] }>(fetch('/api/synthesis/fonts')),
  preset: (name: PresetName) =>
    j<SynthesisPreset>(fetch(`/api/synthesis/preset/${name}`)),

  // Training
  initTraining: (req: {
    architecture: { type: string; params: Record<string, number> }[];
    hyperparameters: TrainingHyperparameters;
    classes: string[];
    synthesis_config?: CheckpointSynthesisConfig;
  }) =>
    j<SessionState & { has_session: true }>(
      fetch('/api/training/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      })
    ),
  trainingState: () => j<SessionState>(fetch('/api/training/state')),
  resetTraining: () =>
    j<{ has_session: false }>(
      fetch('/api/training/reset', { method: 'POST' })
    ),
  predict: (images: string[]) =>
    j<{ predictions: number[][] }>(
      fetch('/api/training/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images }),
      })
    ),
  trainBatch: (
    images: string[],
    labels: string[],
    lr?: number,
    optimizer?: string
  ) =>
    j<TrainBatchResult>(
      fetch('/api/training/train_batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images, labels, lr, optimizer }),
      })
    ),
  evalBatch: (images: string[], labels: string[]) =>
    j<{ loss: number; accuracy: number }>(
      fetch('/api/training/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images, labels }),
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
    j<CheckpointLoadResponse>(
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

  // Inference: rendering + prediction in one call. `chars` is the
  // sequence of glyphs to classify. Each item comes back with both the
  // input PNG (what the model sees) and, when a session is loaded, the
  // model's top prediction re-rendered the same way.
  inferenceRender: (chars: string[], fonts?: string[]) =>
    j<{
      items: InferenceItem[];
      has_session: boolean;
    }>(
      fetch('/api/inference/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chars, fonts }),
      })
    ),

  // ── Pipeline-state mirror ───────────────────────────────────────────
  getState: () => j<PipelineStateSnapshot>(fetch('/api/state')),
  patchState: (patch: Record<string, unknown>) =>
    j<{ ok: true }>(
      fetch('/api/state/patch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch }),
      })
    ),

  // ── ML Engineer chat ────────────────────────────────────────────────
  listChatSessions: (limit = 50) =>
    j<{ sessions: ChatSessionInfo[] }>(
      fetch(`/api/agent/sessions?limit=${limit}`)
    ),
  loadChatSession: (sessionId: string) =>
    j<{ session_id: string; events: ChatEvent[] }>(
      fetch(
        `/api/agent/sessions/${encodeURIComponent(sessionId)}/messages`
      )
    ),
  stopChat: () =>
    j<{ cancelled: boolean; reason?: string }>(
      fetch('/api/agent/stop', { method: 'POST' })
    ),
};

// ── Chat / state types ─────────────────────────────────────────────────

export type PipelineStateSnapshot = {
  synthesis: {
    selectedCategories: Record<string, boolean>;
    fontUsage: Record<string, 'off' | 'train' | 'val'>;
    augmentation: {
      noise: { enabled: boolean; max_level: number };
      skew: { enabled: boolean };
    };
    activePreset: string | null;
  };
  architecture: {
    layers: { type: string; params: Record<string, number> }[];
    hyperparameters: { lr: number; batch_size: number; optimizer: string };
  };
  training: { validateEveryN: number; samplesPerSymbolPerEpoch: number };
};

export type ChatSessionInfo = {
  session_id: string;
  summary: string | null;
  message_count: number | null;
  last_modified: number | null;
  created_at: number | null;
};

// One event from the agent runtime — same shape used both by the live
// SSE stream and by GET /api/agent/sessions/{id}/messages.
export type ChatEvent =
  | { type: 'text_delta'; text: string }
  | { type: 'text_message'; text: string }
  | { type: 'user_message'; text: string }
  | {
      type: 'tool_use';
      id: string;
      name: string;
      full_name: string;
      input: unknown;
    }
  | {
      type: 'tool_result';
      tool_use_id: string;
      content: string;
      is_error: boolean;
    }
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

// State broadcast events (sent over /ws/state by the backend).
//
// `state_replace` / `state_patch` cover the deep-merge pipeline_state
// mirror (synthesis / architecture / training prefs).
//
// The remaining variants are agent-side training progress signals: they
// don't touch pipeline_state, but they do need to flow into the
// frontend's `training` $state so loss charts and the step counter
// update live while the agent runs `train_n_batches` etc.
export type StateBroadcastEvent =
  | {
      type: 'state_replace';
      source: string;
      state: PipelineStateSnapshot;
    }
  | {
      type: 'state_patch';
      source: string;
      patch: Record<string, unknown>;
    }
  | {
      type: 'training_session';
      source: string;
      hasSession: boolean;
      numClasses?: number;
      paramCount?: number;
      step?: number;
      lossHistory?: LossPoint[];
      valLossHistory?: LossPoint[];
    }
  | {
      type: 'training_tick';
      source: string;
      step: number;
      loss: number;
      accuracy: number;
    }
  | {
      type: 'validation_tick';
      source: string;
      step: number;
      loss: number;
      accuracy: number;
    };

/** Stream agent events from /api/agent/chat as parsed objects. Yields
 *  one event per SSE `data:` frame. Pass `signal` to abort mid-stream
 *  (the backend's running turn is cancelled by /api/agent/stop). */
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
      // SSE frames are separated by blank lines. Each frame can have
      // multiple `data:` lines; we concatenate them into one JSON blob.
      let sep: number;
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLines = frame
          .split('\n')
          .filter((l) => l.startsWith('data:'))
          .map((l) => l.slice(5).trimStart());
        if (dataLines.length === 0) continue;
        const payload = dataLines.join('\n');
        try {
          yield JSON.parse(payload) as ChatEvent;
        } catch {
          // Malformed frame — skip rather than tear down the stream.
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Open a WebSocket to /ws/state and yield parsed events. The yielded
 *  generator's `return()` (called when the consumer breaks out) closes
 *  the socket. Reconnect is the caller's responsibility. */
export function openStateSocket(
  onEvent: (ev: StateBroadcastEvent) => void,
  onClose?: (e: CloseEvent) => void,
): WebSocket {
  // Browser WS doesn't honor relative paths — build the URL explicitly.
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${window.location.host}/ws/state`;
  const ws = new WebSocket(url);
  ws.addEventListener('message', (ev) => {
    try {
      const data = JSON.parse(ev.data) as StateBroadcastEvent;
      onEvent(data);
    } catch {
      // Malformed message — skip.
    }
  });
  if (onClose) ws.addEventListener('close', onClose);
  return ws;
}

export type InferencePrediction = {
  char: string;
  png_b64: string | null;
  font: string | null;
  confidence: number;
};

export type InferenceItem = {
  char: string;
  input_png_b64: string | null;
  input_font: string | null;
  predicted_char: string | null;
  predicted_png_b64: string | null;
  predicted_font: string | null;
  confidence: number | null;
  in_class_set: boolean;
  top_k: InferencePrediction[];
};

export type SynthesisSample = {
  png_b64: string;
  label: string;
  font: string;
  missing_glyph: boolean;
};

export type SampleRequest = {
  categories: string[];
  training_fonts: string[];
  validation_fonts: string[];
  augmentation: {
    noise: { enabled: boolean; max_level: number };
    skew: { enabled: boolean };
  };
  split: 'train' | 'val';
  count: number;
  seed: number;
};

/** Stream synthesized samples as NDJSON. Yields one sample at a time as it
 *  arrives over the wire. Pass an `AbortSignal` to cancel cleanly when the
 *  consumer (e.g. a closed modal) is gone. */
export async function* streamSamples(
  req: SampleRequest,
  signal?: AbortSignal
): AsyncGenerator<SynthesisSample> {
  const response = await fetch('/api/synthesis/sample', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(detail || `HTTP ${response.status}`);
  }
  if (!response.body) {
    throw new Error('streaming response has no body');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) yield JSON.parse(trimmed) as SynthesisSample;
      }
    }
    const tail = buffer.trim();
    if (tail) yield JSON.parse(tail) as SynthesisSample;
  } finally {
    reader.releaseLock();
  }
}

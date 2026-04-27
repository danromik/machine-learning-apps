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
    j<{ files: string[] }>(fetch('/api/training/checkpoints')),
  saveCheckpoint: (filename: string) =>
    j<{ name: string }>(
      fetch('/api/training/checkpoints/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      })
    ),
  loadCheckpoint: (filename: string) =>
    j<SessionState & { has_session: true }>(
      fetch('/api/training/checkpoints/load', {
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
};

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

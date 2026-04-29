async function j<T>(p: Promise<Response>): Promise<T> {
  const res = await p;
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Dataset ─────────────────────────────────────────────────────────────

export type ClassEntry = {
  index: number;
  wnid: string;
  label: string;
};

export type PerClassCount = {
  wnid: string;
  label: string;
  train: number;
  val: number;
};

export type DatasetStatus = {
  archive_present: boolean;
  archive_size: number;
  extracted: boolean;
  extract_size: number;
  data_dir: string;
  url: string;
  num_train: number;
  num_val: number;
  per_class: PerClassCount[];
};

export type Sample = {
  png_b64: string;
  label: string;
  label_index: number;
  source: string;
};

export type SampleRequest = {
  split: 'train' | 'val';
  count: number;
  seed: number;
  flip?: boolean;
  jitter?: number;
  random_crop?: boolean;
};

// Snapshot of the data acquisition pipeline saved alongside model weights.
export type DatasetConfig = {
  augmentation: {
    flip: boolean;
    jitter: number;
    random_crop: boolean;
  };
};

// ── Architecture presets ───────────────────────────────────────────────

export type ArchitectureLayerSpec = {
  type: string;
  params: Record<string, number>;
};

export type ArchitecturePreset = {
  name: 'lenet5' | 'alexnet' | 'resnet18';
  label: string;
  year: number;
  tagline: string;
  description: string;
  layers: ArchitectureLayerSpec[];
  locked: boolean;
  hyperparameters: TrainingHyperparameters;
};

// ── Training ────────────────────────────────────────────────────────────

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
      preset: string | null;
    };

export type CheckpointLayer = {
  type: string;
  params: Record<string, number>;
};

export type LossPoint = { step: number; loss: number };

export type CheckpointFile = {
  name: string;
  size: number;
  mtime: number;
};

export type CheckpointLoadResponse = {
  has_session: true;
  num_classes: number;
  param_count: number;
  step: number;
  lr: number;
  batch_size: number;
  optimizer: string;
  preset: string | null;
  layers: CheckpointLayer[];
  classes: string[];
  dataset_config: DatasetConfig | null;
  loss_history: LossPoint[];
  val_loss_history: LossPoint[];
};

export type TrainBatchResult = { loss: number; step: number; accuracy: number };

// ── Devices ─────────────────────────────────────────────────────────────

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

// ── Inference ───────────────────────────────────────────────────────────

export type InferencePrediction = {
  label: string;
  index: number;
  confidence: number;
};

export type InferenceItem = {
  input_png_b64: string;
  predicted_label: string | null;
  confidence: number | null;
  top_k: InferencePrediction[];
  has_session: boolean;
  // Only set when the image came from the val set (not a user upload).
  true_label?: string;
  source?: string;
};

// ── API ────────────────────────────────────────────────────────────────

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

  // Dataset
  classes: () =>
    j<{ classes: ClassEntry[]; input_size: number }>(
      fetch('/api/dataset/classes')
    ),
  datasetStatus: () => j<DatasetStatus>(fetch('/api/dataset/status')),
  downloadDataset: () =>
    j<DatasetStatus>(fetch('/api/dataset/download', { method: 'POST' })),

  // Architecture
  architecturePresets: () =>
    j<{ presets: ArchitecturePreset[] }>(fetch('/api/architecture/presets')),

  // Training
  initTraining: (req: {
    architecture: ArchitectureLayerSpec[];
    preset: string | null;
    hyperparameters: TrainingHyperparameters;
    classes: string[];
    dataset_config?: DatasetConfig;
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
    j<{ has_session: false }>(fetch('/api/training/reset', { method: 'POST' })),
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

  // Inference: classify a user-uploaded image (multipart) or pick a
  // random val image.
  inferenceUpload: (file: File) => {
    const fd = new FormData();
    fd.append('file', file);
    return j<InferenceItem>(
      fetch('/api/inference/predict', {
        method: 'POST',
        body: fd,
      })
    );
  },
  inferenceSample: (seed?: number) =>
    j<InferenceItem>(
      fetch('/api/inference/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(seed === undefined ? {} : { seed }),
      })
    ),
};

/** Stream a dataset batch as NDJSON. Yields one sample at a time. */
export async function* streamSamples(
  req: SampleRequest,
  signal?: AbortSignal
): AsyncGenerator<Sample> {
  const response = await fetch('/api/dataset/sample_stream', {
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
        if (trimmed) yield JSON.parse(trimmed) as Sample;
      }
    }
    const tail = buffer.trim();
    if (tail) yield JSON.parse(tail) as Sample;
  } finally {
    reader.releaseLock();
  }
}

/** Stream the dataset download progress as NDJSON. */
export type DownloadEvent =
  | { stage: 'download' | 'extract'; downloaded: number; total: number; fraction: number }
  | { done: true; status: DatasetStatus }
  | { error: string };

export async function* streamDownload(
  signal?: AbortSignal
): AsyncGenerator<DownloadEvent> {
  const response = await fetch('/api/dataset/download_stream', {
    method: 'POST',
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
        if (trimmed) yield JSON.parse(trimmed) as DownloadEvent;
      }
    }
    const tail = buffer.trim();
    if (tail) yield JSON.parse(tail) as DownloadEvent;
  } finally {
    reader.releaseLock();
  }
}

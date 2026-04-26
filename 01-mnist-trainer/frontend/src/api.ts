export type Sample = { png_b64: string; label: number };
export type Counts = { train: number; test: number };
export type PredictResult = {
  pred: number;
  probs: number[];
  preview_b64: string;
  ckpt_name: string | null;
};
export type TrainStartReq = {
  model: string;
  epochs: number;
  batch_size: number;
  lr: number;
  seed: number;
  max_steps?: number | null;
  max_epochs?: number | null;
};

async function j<T>(p: Promise<Response>): Promise<T> {
  const res = await p;
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export type LayerSpec = {
  type:
    | 'input'
    | 'output'
    | 'linear'
    | 'conv2d'
    | 'maxpool2d'
    | 'flatten'
    | 'activation'
    | 'dropout'
    | (string & {});
  label: string;
  shape: number[];
  size: number;
  details?: Record<string, unknown>;
};

export type Architecture = { name: string; layers: LayerSpec[] };

export const api = {
  device: () => j<{ device: string }>(fetch('/api/device')),
  params: (model: string) =>
    j<{ params: number }>(fetch(`/api/params?model=${model}`)),
  architecture: (model: string) =>
    j<Architecture>(fetch(`/api/architecture?model=${model}`)),
  counts: (cls: number | null) =>
    j<Counts>(
      fetch(`/api/counts${cls == null ? '' : `?class_filter=${cls}`}`)
    ),
  sample: (split: string, cls: string, order: string, n = 200, offset = 0) =>
    j<{ samples: Sample[]; offset: number; total: number }>(
      fetch(
        `/api/sample_q?split=${split}&cls=${cls}&order=${order}&n=${n}&offset=${offset}`
      )
    ),
  checkpoints: () =>
    j<{ files: string[]; current: string | null }>(fetch('/api/checkpoints')),
  session: () =>
    j<{
      has_session: boolean;
      model: string | null;
      step: number;
      epoch: number;
      best_acc: number;
      running: boolean;
    }>(fetch('/api/session')),
  loadCheckpoint: (
    name: string,
    cfg?: { epochs: number; batch_size: number; lr: number; seed: number }
  ) =>
    j<{ name: string; model: string; step: number; epoch: number; best_acc: number }>(
      fetch('/api/checkpoints/load', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name, cfg }),
      })
    ),
  saveCheckpoint: () =>
    j<{ name: string }>(fetch('/api/checkpoints/save', { method: 'POST' })),
  trainStart: (req: TrainStartReq) =>
    j<{ ok: boolean }>(
      fetch('/api/train/start', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(req),
      })
    ),
  trainStop: () =>
    j<{ ok: boolean }>(fetch('/api/train/stop', { method: 'POST' })),
  resetSession: (cfg: {
    model: string;
    epochs: number;
    batch_size: number;
    lr: number;
    seed: number;
  }) =>
    j<{ ok: boolean }>(
      fetch('/api/session/reset', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(cfg),
      })
    ),
  autosave: (enabled: boolean) =>
    j<{ enabled: boolean }>(
      fetch('/api/autosave', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
    ),
  predict: (data_url: string) =>
    j<PredictResult>(
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ data_url }),
      })
    ),
};

export type TrainEvent =
  | { type: 'ping' }
  | { type: 'reset' }
  | { type: 'log'; msg: string }
  | {
      type: 'start';
      steps_per_epoch: number;
      total_steps: number;
      epochs: number;
      starting_step: number;
      starting_epoch: number;
      max_steps: number | null;
      max_epochs: number | null;
    }
  | { type: 'step'; step: number; epoch: number; train_loss: number }
  | {
      type: 'epoch';
      step: number;
      epoch: number;
      val_loss: number;
      val_acc: number;
      seconds: number;
    }
  | { type: 'checkpoint'; name: string; val_acc: number }
  | { type: 'paused'; step: number; epoch: number }
  | { type: 'done'; best_acc: number }
  | { type: 'stopped' }
  | { type: 'error'; msg: string };

export function openTrainSocket(onEvent: (ev: TrainEvent) => void) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {
      /* ignore malformed */
    }
  };
  return ws;
}

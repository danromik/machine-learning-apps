import type { Sample } from './api';

export const cfg = $state({
  model: 'cnn' as 'cnn' | 'mlp',
  epochs: 3,
  batch_size: 128,
  lr: 0.001,
  seed: 0,
});

export const ui = $state({
  device: '',
  params: 0,
  checkpointBadge: '(none)',
  status: 'idle',
  training: false,
  isContinuous: false,
  autoSave: true,
  checkpoints: [] as string[],
  selectedCheckpoint: '' as string,
  classPred: null as number | null,
  predProbs: Array<number>(10).fill(0),
  predPreviewB64: '',
  cycles: 0, // persistent training-step counter
  // Bumped by Data Explorer when the user clicks a sample, asking the
  // drawing canvas to load that image and classify it.
  loadImage: null as null | { png_b64: string; label: number; seq: number },
});

// Bumped when the active theme changes, so chart components can rebuild.
export const theme = $state({ version: 0 });

export const explorer = $state({
  split: 'train',
  cls: 'all',
  order: 'default',
  samples: [] as Sample[],
  trainCount: 0,
  testCount: 0,
  loading: false,
});

export const chartData = $state({
  steps: [] as number[],
  losses: [] as number[],
  epochs: [] as number[],
  valLosses: [] as number[],
  valAccs: [] as number[],
});

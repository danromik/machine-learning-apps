// Layer types, templates, and pure shape/param computation for the
// architecture tab. Mirrors the layer vocabulary used by 01-mnist-trainer
// (conv2d, maxpool2d, flatten, linear, relu, dropout) so the backend can
// later instantiate a torch model from this same shape.

export type LayerType =
  | 'conv2d'
  | 'maxpool2d'
  | 'flatten'
  | 'linear'
  | 'relu'
  | 'dropout';

export type Layer = {
  id: string;
  type: LayerType;
  params: Record<string, number>;
};

export type LayerTemplate = {
  label: string;
  description: string;
  defaults: Record<string, number>;
  primary?: { key: string; format?: (v: number) => string };
};

export const LAYER_TEMPLATES: Record<LayerType, LayerTemplate> = {
  conv2d: {
    label: 'Conv2D',
    description: '2D convolution',
    defaults: { out_channels: 32, kernel: 3, padding: 1, stride: 1 },
    primary: { key: 'out_channels' },
  },
  maxpool2d: {
    label: 'MaxPool',
    description: '2D max pooling (downsampling)',
    defaults: { kernel: 2, stride: 2 },
    primary: { key: 'kernel', format: (v) => `${v}×${v}` },
  },
  flatten: {
    label: 'Flatten',
    description: 'Reshape to 1D vector',
    defaults: {},
  },
  linear: {
    label: 'Linear',
    description: 'Fully-connected (dense) layer',
    defaults: { out_features: 128 },
    primary: { key: 'out_features' },
  },
  relu: {
    label: 'ReLU',
    description: 'Rectified-linear activation',
    defaults: {},
  },
  dropout: {
    label: 'Dropout',
    description: 'Randomly zero a fraction of activations',
    defaults: { p: 0.25 },
    primary: { key: 'p' },
  },
};

export const LAYER_ORDER: LayerType[] = [
  'conv2d',
  'maxpool2d',
  'flatten',
  'linear',
  'relu',
  'dropout',
];

export type LayerComputed = {
  shape: number[] | null;
  params: number;
  error?: string;
};

export type ArchitectureComputed = {
  // shapes[0] = input, shapes[i+1] = after user layer i, shapes[N+1] = output
  shapes: (number[] | null)[];
  perLayerParams: number[];
  errors: (string | undefined)[];
  totalParams: number;
  // True iff the implicit output Linear's input is 1D (i.e., user ended with
  // a vector-shaped chain so the final classifier can be projected).
  outputValid: boolean;
};

export function applyLayer(layer: Layer, shape: number[] | null): LayerComputed {
  if (shape === null) return { shape: null, params: 0, error: 'previous shape unknown' };

  switch (layer.type) {
    case 'conv2d': {
      if (shape.length !== 3) {
        return { shape: null, params: 0, error: 'Conv2D needs 3D input (C×H×W)' };
      }
      const [c, h, w] = shape;
      const oc = Number(layer.params.out_channels);
      const k = Number(layer.params.kernel);
      const p = Number(layer.params.padding ?? 0);
      const s = Number(layer.params.stride ?? 1);
      const oh = Math.floor((h + 2 * p - k) / s) + 1;
      const ow = Math.floor((w + 2 * p - k) / s) + 1;
      if (oh <= 0 || ow <= 0) {
        return { shape: null, params: 0, error: 'Conv output shrinks below 1×1' };
      }
      return { shape: [oc, oh, ow], params: oc * (c * k * k + 1) };
    }
    case 'maxpool2d': {
      if (shape.length !== 3) {
        return { shape: null, params: 0, error: 'MaxPool needs 3D input' };
      }
      const [c, h, w] = shape;
      const k = Number(layer.params.kernel);
      const s = Number(layer.params.stride ?? k);
      const oh = Math.floor((h - k) / s) + 1;
      const ow = Math.floor((w - k) / s) + 1;
      if (oh <= 0 || ow <= 0) {
        return { shape: null, params: 0, error: 'Pool output shrinks below 1×1' };
      }
      return { shape: [c, oh, ow], params: 0 };
    }
    case 'flatten': {
      const total = shape.reduce((a, b) => a * b, 1);
      return { shape: [total], params: 0 };
    }
    case 'linear': {
      if (shape.length !== 1) {
        return {
          shape: null,
          params: 0,
          error: 'Linear needs 1D input — add Flatten first',
        };
      }
      const inF = shape[0];
      const outF = Number(layer.params.out_features);
      return { shape: [outF], params: inF * outF + outF };
    }
    case 'relu':
    case 'dropout':
      return { shape, params: 0 };
  }
}

export function computeArchitecture(
  layers: Layer[],
  inputShape: number[],
  numClasses: number
): ArchitectureComputed {
  const shapes: (number[] | null)[] = [inputShape];
  const perLayerParams: number[] = [];
  const errors: (string | undefined)[] = [];
  let totalParams = 0;

  let curShape: number[] | null = inputShape;

  for (const layer of layers) {
    const r = applyLayer(layer, curShape);
    shapes.push(r.shape);
    perLayerParams.push(r.params);
    errors.push(r.error);
    totalParams += r.params;
    curShape = r.shape;
  }

  // Implicit final classifier: project the user's vector to numClasses.
  // Counted in totalParams but not shown as an explicit layer block — the
  // Output block represents this projection plus softmax.
  const outputValid = curShape !== null && curShape.length === 1;
  if (outputValid && curShape) {
    totalParams += curShape[0] * numClasses + numClasses;
  }
  shapes.push([numClasses]);

  return {
    shapes,
    perLayerParams,
    errors,
    totalParams,
    outputValid,
  };
}

// Compact human-readable count (e.g., 1.23M, 7.4K, 982).
export function formatCount(n: number): string {
  if (!Number.isFinite(n)) return '–';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}


// ── Architecture suggestion ─────────────────────────────────────────────

export type SuggestInput = {
  numClasses: number;
  numTrainFonts: number;
  noiseEnabled: boolean;
  noiseLevel: number; // 0–100
  skewEnabled: boolean;
};

export type ArchitectureSuggestion = {
  layers: Layer[];
  reasoning: string;
};

/** Heuristic for the depth/width of a CNN classifier given the synthesis
 *  config. Classes drive the FC head width; classes + fonts + augmentation
 *  jointly drive the number of conv blocks. */
export function suggestArchitecture(opts: SuggestInput): ArchitectureSuggestion {
  const { numClasses, numTrainFonts, noiseEnabled, noiseLevel, skewEnabled } = opts;

  // Complexity score in roughly the same units as log2(classes).
  // - log2(classes) is the dominant term (10 classes → 3.3, 184 → 7.5).
  // - log2(fonts) at half weight reflects that font diversity matters less
  //   than the raw number of labels.
  // - Augmentation adds a flat bump (heavy noise + skew → ~+0.8).
  const augBump =
    (noiseEnabled ? noiseLevel / 100 : 0) + (skewEnabled ? 0.3 : 0);
  const complexity =
    Math.log2(Math.max(2, numClasses)) +
    0.5 * Math.log2(Math.max(2, numTrainFonts)) +
    augBump;

  // Tier the network at thresholds chosen so that the three Data Synthesis
  // presets (Beginner, Intermediate, Advanced) land in tiers 1, 2, 3.
  let convBlocks: number;
  let fcWidth: number;
  let tierName: string;
  if (complexity < 5) {
    convBlocks = 2;
    fcWidth = 128;
    tierName = 'small';
  } else if (complexity < 8) {
    convBlocks = 3;
    fcWidth = 256;
    tierName = 'medium';
  } else {
    convBlocks = 4;
    fcWidth = 512;
    tierName = 'large';
  }

  // Channels: 32, 64, 128, then plateau at 128 (the 64×64 input only
  // supports 4 halving-pools cleanly — 64 → 32 → 16 → 8 → 4).
  const channels: number[] = [];
  for (let i = 0; i < convBlocks; i++) {
    channels.push(32 << Math.min(i, 2));
  }

  // Build layer list.
  const layers: Layer[] = [];
  let counter = 0;
  const nid = () => `s${Date.now().toString(36)}-${counter++}`;

  for (const ch of channels) {
    layers.push({
      id: nid(),
      type: 'conv2d',
      params: { out_channels: ch, kernel: 3, padding: 1, stride: 1 },
    });
    layers.push({ id: nid(), type: 'relu', params: {} });
    layers.push({
      id: nid(),
      type: 'maxpool2d',
      params: { kernel: 2, stride: 2 },
    });
  }
  layers.push({ id: nid(), type: 'flatten', params: {} });
  layers.push({
    id: nid(),
    type: 'linear',
    params: { out_features: fcWidth },
  });
  layers.push({ id: nid(), type: 'relu', params: {} });
  layers.push({ id: nid(), type: 'dropout', params: { p: 0.25 } });

  // Reasoning text — keep it short but informative.
  const augDesc =
    [
      noiseEnabled && `${noiseLevel}% noise`,
      skewEnabled && 'skew',
    ]
      .filter(Boolean)
      .join(' + ') || 'no augmentation';
  const channelStr = channels.join('→');
  const fcRatio = (fcWidth / Math.max(1, numClasses)).toFixed(1);

  const reasoning =
    `Suggested ${tierName} CNN for ${numClasses} class${numClasses === 1 ? '' : 'es'} ` +
    `× ${numTrainFonts} training font${numTrainFonts === 1 ? '' : 's'} (${augDesc}). ` +
    `${convBlocks} conv blocks (3×3 conv → ReLU → 2×2 pool, channels ${channelStr}) ` +
    `build hierarchical visual features — more blocks for richer font/augmentation variety. ` +
    `FC head with ${fcWidth} units (≈ ${fcRatio}× class count) gives the classifier enough ` +
    `capacity, with dropout 0.25 to regularize.`;

  return { layers, reasoning };
}

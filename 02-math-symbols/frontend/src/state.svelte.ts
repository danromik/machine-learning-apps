import type { Font, PresetName, SymbolCategory, SynthesisPreset } from './api';
import type { Layer, LayerType } from './components/tabs/architecture/computeArchitecture';

export type TabId = 'orientation' | 'data' | 'architecture' | 'training' | 'inference';

export const ui = $state({
  activeTab: 'orientation' as TabId,
  device: '' as string,
  status: 'idle' as string,
});

// Bumped when the active theme changes, so chart components can rebuild
// against the new CSS variables. (Will be used once charts land in the
// Training tab — kept here so cross-tab state has one home.)
export const theme = $state({ version: 0 });

// ── Data Synthesis state ────────────────────────────────────────────────

export type FontUsage = 'off' | 'train' | 'val';

export const synthesis = $state({
  loaded: false,
  categories: [] as SymbolCategory[],
  fonts: [] as Font[],
  selectedCategories: {} as Record<string, boolean>,
  fontUsage: {} as Record<string, FontUsage>,
  augmentation: {
    noise: { enabled: false, max_level: 25 },
    skew: { enabled: false },
  },
  // The preset most recently applied. Cleared when the user manually
  // modifies any field — that's how the segmented control's active state
  // toggles off when the config drifts away from a named preset.
  activePreset: null as PresetName | null,
  // Per-section disclosure state. Persists across tab switches so the
  // user's expanded/collapsed choices don't reset every time they leave
  // the data tab and come back.
  collapsed: {
    symbols: false,
    fonts: false,
    augmentation: false,
  },
});

export function applyPreset(preset: SynthesisPreset) {
  const cats: Record<string, boolean> = {};
  for (const c of synthesis.categories) {
    cats[c.id] = preset.categories.includes(c.id);
  }
  synthesis.selectedCategories = cats;

  const trainSet = new Set(preset.training_fonts);
  const valSet = new Set(preset.validation_fonts);
  const fonts: Record<string, FontUsage> = {};
  for (const f of synthesis.fonts) {
    if (trainSet.has(f.family)) fonts[f.family] = 'train';
    else if (valSet.has(f.family)) fonts[f.family] = 'val';
    else fonts[f.family] = 'off';
  }
  synthesis.fontUsage = fonts;

  synthesis.augmentation = {
    noise: { ...preset.augmentation.noise },
    skew: { ...preset.augmentation.skew },
  };
  synthesis.activePreset = preset.name;
}

export function markModified() {
  synthesis.activePreset = null;
}

// ── Architecture state ──────────────────────────────────────────────────

export type Optimizer = 'adam' | 'adamw' | 'sgd';

export const architecture = $state({
  layers: [] as Layer[],
  hyperparameters: {
    lr: 0.001,
    batch_size: 128,
    optimizer: 'adam' as Optimizer,
  },
  // Last suggestion's reasoning string, shown next to the Suggest button.
  // Persists across tab switches; cleared (or replaced) only on the next
  // Suggest click. Survives manual edits — the explanation describes what
  // the last suggestion *was*, not what's currently on the diagram.
  suggestionReasoning: null as string | null,
});

// Image dimensions assumed for now — will become a Data Synthesis option.
// 1×64×64 grayscale matches the working size for printed-glyph OCR.
export const INPUT_SHAPE: number[] = [1, 64, 64];

// Cross-component drag state. Set on dragstart by LayerSidebar so the
// diagram can render the right '+ TypeName' indicator while the drag is
// in flight; cleared on dragend (which fires reliably on the source even
// when the drop is cancelled outside any drop target).
export const dragState = $state({
  draggingType: null as LayerType | null,
});

// ── Training state ─────────────────────────────────────────────────────

import type { SynthesisSample } from './api';

export const training = $state({
  // Session info mirrored from the backend.
  hasSession: false,
  numClasses: 0,
  paramCount: 0,
  step: 0,

  // Currently-loaded batch (size = architecture.hyperparameters.batch_size).
  batch: [] as SynthesisSample[],
  // Softmax probs per batch item (parallel to batch). Empty until a session
  // is initialized and predict has been called.
  predictions: [] as number[][],
  // Index of the highlighted image in the batch (manual click or animation).
  selectedIndex: null as number | null,
  // True while Train (1 Batch) is animating through images.
  animating: false,
  // Per-batch-item verdict from the Train 1 Batch (Fun) sweep — null until
  // the animation reaches that index, then 'correct' or 'incorrect' for
  // the rest of the batch's lifetime. Cleared when a new batch loads so
  // the user gets a fresh canvas.
  batchVerdict: [] as ('correct' | 'incorrect' | null)[],
  // Persistent counts powering the BatchChart. Survives loadBatch so the
  // last training run's tally stays on screen until another run replaces
  // it. `total` is the batch size at snapshot time so bars scale right
  // even after batch_size changes between runs.
  batchChartCounts: { correct: 0, incorrect: 0, total: 0 } as {
    correct: number;
    incorrect: number;
    total: number;
  },
  // Last train_batch loss + top-1 accuracy on that batch, for display.
  lastLoss: null as number | null,
  lastAccuracy: null as number | null,

  // An "epoch" here is a number of batches that, on average, exposes each
  // symbol in the target set to N training examples — N is configurable
  // because there's no canonical right value with synthesized data
  // (any number of batches is "valid"; this just sets a meaningful unit).
  samplesPerSymbolPerEpoch: 50,

  // Checkpoint filename text field (sticky across navigations).
  checkpointFilename: 'model.pt',
  // List of available checkpoint files (refreshed on tab mount and after save).
  availableCheckpoints: [] as string[],
});

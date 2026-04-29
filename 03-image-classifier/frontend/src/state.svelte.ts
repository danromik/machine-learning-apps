import { tick } from 'svelte';
import type {
  ArchitecturePreset,
  CheckpointFile,
  CheckpointLoadResponse,
  ClassEntry,
  DatasetConfig,
  DatasetStatus,
  Sample,
} from './api';
import type { Layer, LayerType } from './components/tabs/architecture/computeArchitecture';

export type TabId =
  | 'orientation'
  | 'data'
  | 'architecture'
  | 'training'
  | 'inference'
  | 'debrief';

export const ui = $state({
  activeTab: 'orientation' as TabId,
});

// ── UI prefs persistence ────────────────────────────────────────────────

const UI_PREFS_KEY = 'image-classifier.ui-prefs.v1';

const VALID_TABS: TabId[] = [
  'orientation',
  'data',
  'architecture',
  'training',
  'inference',
  'debrief',
];

function readUiPrefs(): { activeTab?: TabId } {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(UI_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as { activeTab?: unknown };
    if (parsed && typeof parsed === 'object') {
      const t = parsed.activeTab;
      if (typeof t === 'string' && (VALID_TABS as string[]).includes(t)) {
        return { activeTab: t as TabId };
      }
    }
    return {};
  } catch {
    return {};
  }
}

export function persistUiPrefs() {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(
      UI_PREFS_KEY,
      JSON.stringify({ activeTab: ui.activeTab })
    );
  } catch {}
}

const _initialUiPrefs = readUiPrefs();
if (_initialUiPrefs.activeTab) {
  ui.activeTab = _initialUiPrefs.activeTab;
}

export const theme = $state({ version: 0 });

// ── Dataset state ───────────────────────────────────────────────────────
//
// The Image Classifier's data layer is much simpler than Math Symbols':
// the class set is fixed (10 Imagenette categories), the dataset is a
// known download (no per-symbol opt-in), and train/val are pre-split by
// fast.ai. The "dataset" store therefore only tracks:
//   • whether the dataset has been downloaded;
//   • per-class counts on disk;
//   • augmentation knobs the user can experiment with.

export const dataset = $state({
  loaded: false,                          // classes + status fetched
  classes: [] as ClassEntry[],
  inputSize: 96,
  status: null as DatasetStatus | null,
  // Augmentation toggles — applied to the train pipeline only.
  augmentation: {
    flip: true,
    jitter: 0.0,
    random_crop: true,
  },
  // Set while a download is in flight; the Data Acquisition tab uses
  // these to drive a progress bar.
  downloading: false,
  downloadStage: null as 'download' | 'extract' | null,
  downloadFraction: 0,
  downloadError: null as string | null,
});

/** True when the dataset has been downloaded and extracted on disk. */
export function isDatasetReady(): boolean {
  return dataset.status?.extracted === true && dataset.status.num_train > 0;
}

/** Just the class labels, in index order — the canonical class table the
 *  backend uses to map labels ↔ indices. */
export function classLabels(): string[] {
  return dataset.classes.map((c) => c.label);
}

// ── Architecture state ──────────────────────────────────────────────────

export type Optimizer = 'adam' | 'adamw' | 'sgd';

export const architecture = $state({
  // Preset name when the architecture is one of the named presets,
  // null when the user is freeform-building. ResNet-18 is the only
  // *locked* preset: when set, the layer list is empty and the diagram
  // shows a single placeholder block.
  preset: null as ArchitecturePreset['name'] | null,
  layers: [] as Layer[],
  hyperparameters: {
    lr: 0.001,
    batch_size: 64,
    optimizer: 'adam' as Optimizer,
  },
  // Available presets fetched from the backend on startup.
  presets: [] as ArchitecturePreset[],
});

export const INPUT_SHAPE: number[] = [3, 96, 96];

/** Apply a named preset onto the architecture state. Locked presets
 *  (resnet18) flag `preset` and clear the layer list — the diagram shows
 *  a placeholder. Editable presets (lenet5, alexnet) populate `layers`
 *  with fresh-id'd copies the user can tweak afterwards. */
export function applyArchitecturePreset(p: ArchitecturePreset) {
  let counter = 0;
  const nid = () => `p${Date.now().toString(36)}-${counter++}`;
  architecture.preset = p.name;
  architecture.layers = p.locked
    ? []
    : p.layers.map((l) => ({
        id: nid(),
        type: l.type as LayerType,
        params: { ...l.params },
      }));
  architecture.hyperparameters = {
    lr: p.hyperparameters.lr,
    batch_size: p.hyperparameters.batch_size,
    optimizer: p.hyperparameters.optimizer as Optimizer,
  };
}

export function clearPreset() {
  architecture.preset = null;
}

export const dragState = $state({
  draggingType: null as LayerType | null,
});

// ── Training state ──────────────────────────────────────────────────────

export const training = $state({
  hasSession: false,
  numClasses: 0,
  paramCount: 0,
  step: 0,
  preset: null as string | null,

  batch: [] as Sample[],
  predictions: [] as number[][],
  selectedIndex: null as number | null,
  animating: false,
  batchVerdict: [] as ('correct' | 'incorrect' | null)[],
  batchChartCounts: { correct: 0, incorrect: 0, total: 0 } as {
    correct: number;
    incorrect: number;
    total: number;
  },

  lossHistory: [] as { step: number; loss: number }[],
  valLossHistory: [] as { step: number; loss: number }[],
  validateEveryN: 10,
  lastLoss: null as number | null,
  lastAccuracy: null as number | null,

  busy: false,
  statusMsg: 'idle' as string,
  epochRunning: false,
  abortEpoch: false,
  continuousRunning: false,
  abortContinuous: false,

  checkpointFilename: 'model.pt',
  availableCheckpoints: [] as CheckpointFile[],
  autoSave: false,
  autoLoadOnRestart: false,
  // While > 0, App.svelte's dataset-change effect skips wholesale reset.
  suppressDatasetInvalidation: 0,
});

// ── Checkpoint UI prefs persistence ─────────────────────────────────────

const CKPT_PREFS_KEY = 'image-classifier.checkpoint-prefs.v1';

type CheckpointPrefs = {
  filename: string;
  autoSave: boolean;
  autoLoadOnRestart: boolean;
};

function readCheckpointPrefs(): Partial<CheckpointPrefs> {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(CKPT_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<CheckpointPrefs>;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function persistCheckpointPrefs() {
  if (typeof localStorage === 'undefined') return;
  const prefs: CheckpointPrefs = {
    filename: training.checkpointFilename,
    autoSave: training.autoSave,
    autoLoadOnRestart: training.autoLoadOnRestart,
  };
  try {
    localStorage.setItem(CKPT_PREFS_KEY, JSON.stringify(prefs));
  } catch {}
}

const _initialCkptPrefs = readCheckpointPrefs();
if (typeof _initialCkptPrefs.filename === 'string' && _initialCkptPrefs.filename) {
  training.checkpointFilename = _initialCkptPrefs.filename;
}
if (typeof _initialCkptPrefs.autoSave === 'boolean') {
  training.autoSave = _initialCkptPrefs.autoSave;
}
if (typeof _initialCkptPrefs.autoLoadOnRestart === 'boolean') {
  training.autoLoadOnRestart = _initialCkptPrefs.autoLoadOnRestart;
}

// ── Checkpoint response → state restoration ─────────────────────────────

export async function applyCheckpointResponse(r: CheckpointLoadResponse) {
  // Dataset config: only assign augmentation when the checkpoint carries it.
  if (r.dataset_config) {
    training.suppressDatasetInvalidation++;
    try {
      dataset.augmentation = {
        flip: r.dataset_config.augmentation.flip,
        jitter: r.dataset_config.augmentation.jitter,
        random_crop: r.dataset_config.augmentation.random_crop,
      };
      await tick();
    } finally {
      training.suppressDatasetInvalidation--;
    }
  }

  // Architecture: preset takes precedence over the raw layer list.
  let counter = 0;
  const nid = () => `ckpt-${Date.now().toString(36)}-${counter++}`;
  architecture.preset = (r.preset as ArchitecturePreset['name'] | null) ?? null;
  architecture.layers = r.layers.map((l) => ({
    id: nid(),
    type: l.type as LayerType,
    params: { ...l.params },
  }));
  architecture.hyperparameters = {
    lr: r.lr,
    batch_size: r.batch_size,
    optimizer: r.optimizer as Optimizer,
  };

  training.hasSession = true;
  training.numClasses = r.num_classes;
  training.paramCount = r.param_count;
  training.step = r.step;
  training.preset = r.preset ?? null;
  training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
  training.lossHistory = r.loss_history.map((p) => ({ ...p }));
  training.valLossHistory = r.val_loss_history.map((p) => ({ ...p }));
  const lastTrain =
    training.lossHistory[training.lossHistory.length - 1] ?? null;
  training.lastLoss = lastTrain ? lastTrain.loss : null;
  training.lastAccuracy = null;
  training.batch = [];
  training.predictions = [];
  training.batchVerdict = [];
  training.selectedIndex = null;
}

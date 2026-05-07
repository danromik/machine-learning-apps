import { tick } from 'svelte';
import type {
  CheckpointFile,
  CheckpointLoadResponse,
  Font,
  PresetName,
  SymbolCategory,
  SynthesisPreset,
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
//
// Active tab survives across browser reloads via localStorage. Read once
// at module load (synchronous, before any component mounts) so the user
// lands on whichever tab they were on last. Persisted by an $effect in
// App.svelte that calls persistUiPrefs() on every activeTab change.

const UI_PREFS_KEY = 'agentic-symbols.ui-prefs.v1';

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
  } catch {
    // localStorage can throw under quota / private mode — ignore.
  }
}

const _initialUiPrefs = readUiPrefs();
if (_initialUiPrefs.activeTab) {
  ui.activeTab = _initialUiPrefs.activeTab;
}

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

  // Loss curves for the bottom-right panel. lossHistory gets one point
  // per train_batch step; valLossHistory is sparser (every Nth step,
  // computed via /api/training/eval on a freshly-rendered val batch).
  // Both reset when the model is reinitialized or a checkpoint is
  // loaded — they're only meaningful for the current session.
  lossHistory: [] as { step: number; loss: number }[],
  valLossHistory: [] as { step: number; loss: number }[],
  validateEveryN: 10,
  // Last train_batch loss + top-1 accuracy on that batch, for display.
  lastLoss: null as number | null,
  lastAccuracy: null as number | null,

  // An "epoch" here is a number of batches that, on average, exposes each
  // symbol in the target set to N training examples — N is configurable
  // because there's no canonical right value with synthesized data
  // (any number of batches is "valid"; this just sets a meaningful unit).
  samplesPerSymbolPerEpoch: 50,

  // Live state for in-progress training operations. Lifted out of
  // TrainingTab.svelte so a long-running Train 1 Epoch / Train
  // Continuously loop survives the user navigating to another tab and
  // back — the loop is a JS closure that keeps running, but the local
  // component state would be lost on unmount, leaving the buttons and
  // status footer out of sync with reality on remount.
  busy: false,
  statusMsg: 'idle' as string,
  epochRunning: false,
  abortEpoch: false,
  continuousRunning: false,
  abortContinuous: false,

  // Checkpoint filename text field (sticky across navigations and across
  // browser sessions via localStorage — see initCheckpointPrefs / persist).
  checkpointFilename: 'model.pt',
  // List of available checkpoint files (refreshed on tab mount and after save).
  // Each entry carries name, size in bytes, and mtime in unix seconds so the
  // sidebar can show a "filesize / last save" line under the filename input.
  availableCheckpoints: [] as CheckpointFile[],
  // User toggles for automatic save/load. Persisted to localStorage so they
  // survive page reloads. Auto-save fires after every train op; auto-load
  // fires once at app start when the toggle is on.
  autoSave: false,
  autoLoadOnRestart: false,
  // While > 0, App.svelte's synthesis-change effect skips the wholesale
  // reset (architecture wipe + session drop). Used by the checkpoint load
  // path, which itself rewrites synthesis state to match the loaded model
  // and would otherwise trip the invalidation it's trying to avoid.
  suppressSynthesisInvalidation: 0,
});

// ── Chat (ML Engineer) state ───────────────────────────────────────────
//
// The chat pane on the right-hand side is a multi-turn conversation with
// the ML Engineer agent (Claude Opus 4.7, 1M context). Each user turn
// runs through /api/agent/chat as an SSE stream of typed events; this
// store reduces those events into a flat list of ChatItems the
// transcript renders directly.

export type ChatRole = 'user' | 'assistant';

// One item in the rendered transcript.
//   - text: a chat bubble (user or assistant). Streaming assistant text
//     keeps appending to the last `text` item until a tool call boundary
//     forces a new bubble.
//   - tool: an info row (no bubble) with the tool name, input, status,
//     and (once it returns) the result string. Status flips
//     running → success | error when the matching tool_result arrives.
//   - error: a system-styled error row (e.g. SDK exceptions, cancelled
//     turns).
export type ChatItem =
  | {
      kind: 'text';
      role: ChatRole;
      text: string;
      // True while the assistant is still streaming into this bubble.
      // Closed when a tool_use, error, or result event arrives.
      streaming?: boolean;
    }
  | {
      kind: 'tool';
      id: string;
      name: string;
      input: unknown;
      status: 'running' | 'success' | 'error';
      result?: string;
    }
  | { kind: 'error'; message: string };

export type ChatSessionSummary = {
  session_id: string;
  summary: string | null;
  message_count: number | null;
  last_modified: number | null;
  created_at: number | null;
};

export const CONTEXT_WINDOW = 1_000_000;

// Chat-pane resize bounds. Min keeps the input + transcript usable;
// max keeps the tab content (main column) at least readable. Applied
// during drag and clamped on read of persisted prefs in case the
// window has gotten narrower since the last save.
export const CHAT_PANE_MIN_WIDTH = 240;
export const CHAT_PANE_DEFAULT_WIDTH = 380;
export function clampChatPaneWidth(w: number): number {
  if (typeof window === 'undefined' || !Number.isFinite(w)) {
    return CHAT_PANE_DEFAULT_WIDTH;
  }
  const max = Math.max(CHAT_PANE_MIN_WIDTH, window.innerWidth - 320);
  return Math.max(CHAT_PANE_MIN_WIDTH, Math.min(max, Math.round(w)));
}

export const chat = $state({
  // Pane visibility — toggled by the header button. Persisted to
  // localStorage so the layout survives reloads.
  visible: false,
  // Pane width in px. Live-updated by the divider drag in ChatPane
  // and persisted under chat-prefs so the layout survives reloads.
  paneWidth: CHAT_PANE_DEFAULT_WIDTH,
  // Live transcript — what ChatTranscript renders. Cleared on "New
  // chat" or on session switch.
  items: [] as ChatItem[],
  // Active session id from the SDK. null until the first turn returns,
  // then persisted so subsequent turns resume the same session.
  activeSessionId: null as string | null,
  // Available past sessions, populated by SessionMenu's load.
  sessions: [] as ChatSessionSummary[],
  // Per-turn status. 'idle' between turns, 'streaming' while the SSE
  // stream is open. Drives the input's send/stop button label.
  turn: 'idle' as 'idle' | 'streaming',
  // Cumulative usage from the most recent ResultMessage, plus the
  // session's running cost. Drives the UsageBar.
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    total_cost_usd: 0,
  },
  // Opaque draft text in the input field — kept on the store so it
  // survives ChatPane unmount (e.g. user toggles the pane off and on
  // without losing what they were typing).
  draft: '',
});

// ── Chat prefs persistence ──────────────────────────────────────────────
//
// Two fields ride localStorage: the pane visibility flag and the active
// session id (so a reload returns the user to the conversation they
// were in). Neither should round-trip the session list or transcript —
// those are fetched on demand.

const CHAT_PREFS_KEY = 'agentic-symbols.chat-prefs.v1';

type ChatPrefs = {
  visible: boolean;
  activeSessionId: string | null;
  paneWidth?: number;
};

function readChatPrefs(): Partial<ChatPrefs> {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(CHAT_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Partial<ChatPrefs>;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function persistChatPrefs() {
  if (typeof localStorage === 'undefined') return;
  const prefs: ChatPrefs = {
    visible: chat.visible,
    activeSessionId: chat.activeSessionId,
    paneWidth: chat.paneWidth,
  };
  try {
    localStorage.setItem(CHAT_PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // localStorage can throw under quota / private mode — ignore.
  }
}

const _initialChatPrefs = readChatPrefs();
if (typeof _initialChatPrefs.visible === 'boolean') {
  chat.visible = _initialChatPrefs.visible;
}
if (typeof _initialChatPrefs.activeSessionId === 'string') {
  chat.activeSessionId = _initialChatPrefs.activeSessionId;
}
if (typeof _initialChatPrefs.paneWidth === 'number') {
  // Clamp on read in case the saved width is now wider than the
  // current viewport allows (browser was resized between sessions).
  chat.paneWidth = clampChatPaneWidth(_initialChatPrefs.paneWidth);
}

// ── Per-tab client id ───────────────────────────────────────────────────
//
// State sync echo prevention: when this tab pushes a /api/state/patch,
// the backend broadcasts it back over /ws/state. Without an originator
// id, the WS handler couldn't tell its own edits from the agent's, and
// applying its own echo would feed the auto-sync $effect a fresh diff
// → infinite loop.
//
// One id per tab, generated once at module load. Not persisted — each
// tab needs a unique id so multi-tab edits don't collide.
function _genClientId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export const CLIENT_ID = _genClientId();

// ── Checkpoint UI prefs persistence ─────────────────────────────────────
//
// Three fields ride localStorage: the filename in the textbox, the
// auto-save toggle, and the auto-load-on-restart toggle. We init from
// storage at module load (synchronous, before any component mounts) so
// the textbox and checkboxes render with the user's last values, and the
// auto-load decision can be made the moment the app boots.

const CKPT_PREFS_KEY = 'agentic-symbols.checkpoint-prefs.v1';

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
  } catch {
    // localStorage can throw under quota / private mode — ignore.
  }
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
//
// Apply a /api/training/checkpoints/load response onto the reactive
// stores: synthesis (categories / fonts / augmentation), architecture
// (layers + hyperparameters), and training session metadata. Used by
// the manual Load button and by auto-load-on-restart.
//
// The synthesis-change effect in App.svelte normally reacts to any
// edit by clearing architecture and dropping the backend session.
// That's the wrong behavior here — we *want* the loaded synthesis
// state — so we hold the suppression flag across the synthesis writes
// and tick() to let the effect run-and-skip before releasing it.
export async function applyCheckpointResponse(r: CheckpointLoadResponse) {
  // Synthesis: only assign fields the checkpoint actually carries. An
  // older checkpoint without synthesis_config keeps the user's current
  // synthesis state (and the class-set mismatch will surface on the
  // next train_batch — same behavior we already had pre-feature).
  if (r.synthesis_config) {
    training.suppressSynthesisInvalidation++;
    try {
      synthesis.selectedCategories = { ...r.synthesis_config.selectedCategories };
      synthesis.fontUsage = { ...r.synthesis_config.fontUsage };
      synthesis.augmentation = {
        noise: { ...r.synthesis_config.augmentation.noise },
        skew: { ...r.synthesis_config.augmentation.skew },
      };
      synthesis.activePreset = null;
      // Flush so App.svelte's $effect sees the new sig, takes the
      // suppression branch, and updates lastSynthesisSig — leaving us
      // a clean baseline for future user edits.
      await tick();
    } finally {
      training.suppressSynthesisInvalidation--;
    }
  }

  // Architecture: rebuild layer list from the saved spec, generating
  // fresh frontend IDs. Hyperparameter sliders match the loaded model.
  let counter = 0;
  const nid = () => `ckpt-${Date.now().toString(36)}-${counter++}`;
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
  architecture.suggestionReasoning = null;

  // Training session metadata.
  training.hasSession = true;
  training.numClasses = r.num_classes;
  training.paramCount = r.param_count;
  training.step = r.step;
  training.batchChartCounts = { correct: 0, incorrect: 0, total: 0 };
  // Restore loss curves so charts pick up where the user left off. The
  // backend appends to these on every train_batch / eval_batch, so the
  // checkpoint always carries the full series the user saw on screen.
  training.lossHistory = r.loss_history.map((p) => ({ ...p }));
  training.valLossHistory = r.val_loss_history.map((p) => ({ ...p }));
  // Seed lastLoss / lastAccuracy from the most recent training point so
  // the status badge isn't blank right after load. Accuracy isn't in
  // the saved series — leave it null and let the next batch fill it.
  const lastTrain =
    training.lossHistory[training.lossHistory.length - 1] ?? null;
  training.lastLoss = lastTrain ? lastTrain.loss : null;
  training.lastAccuracy = null;
  // Force a fresh batch on next TrainingTab interaction — the existing
  // batch was rendered against whatever synthesis was set before load.
  training.batch = [];
  training.predictions = [];
  training.batchVerdict = [];
  training.selectedIndex = null;
}

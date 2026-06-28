import type {
  AlgorithmId,
  AlgorithmInfo,
  CheckpointFile,
  IterationRecord,
  RunStatus,
  SessionState,
} from './api';
import { readCubeStyleId } from './cubeStyles';
import { isSoundEnabled } from './sound';

export type TabId =
  | 'orientation'
  | 'cube'
  | 'algorithm'
  | 'training'
  | 'watch'
  | 'progress-report'
  | 'debrief';

export const ui = $state({
  activeTab: 'orientation' as TabId,
});

// ── UI prefs persistence (active tab survives reloads) ──────────────────
const UI_PREFS_KEY = 'agentic-cube.ui-prefs.v1';
const VALID_TABS: TabId[] = [
  'orientation', 'cube', 'algorithm', 'training', 'watch', 'progress-report', 'debrief',
];

function readUiPrefs(): { activeTab?: TabId } {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(UI_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as { activeTab?: unknown };
    const t = parsed?.activeTab;
    if (typeof t === 'string' && (VALID_TABS as string[]).includes(t)) {
      return { activeTab: t as TabId };
    }
    return {};
  } catch {
    return {};
  }
}

export function persistUiPrefs() {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(UI_PREFS_KEY, JSON.stringify({ activeTab: ui.activeTab }));
  } catch {
    // ignore quota / private-mode errors
  }
}

const _initialUiPrefs = readUiPrefs();
if (_initialUiPrefs.activeTab) ui.activeTab = _initialUiPrefs.activeTab;

export const theme = $state({ version: 0 });

// Selected 3D cube color style (see cubeStyles.ts). Persisted; read by CubeView3D.
export const cubeStyle = $state({ id: readCubeStyleId() });

// Master sound on/off (see sound.ts). Persisted; toggled from the status bar.
export const sound = $state({ enabled: isSoundEnabled() });

// ── Cube / environment state ────────────────────────────────────────────
export const cube = $state({
  size: 3 as 2 | 3,
  curriculum: {
    startK: 1,
    maxK: 14,
    promoteAt: 0.9,
  },
});

// ── Algorithm state ─────────────────────────────────────────────────────
export const algorithm = $state({
  loaded: false,
  algo: 'value_iteration' as AlgorithmId,
  hyperparameters: {} as Record<string, number>,
  catalog: [] as AlgorithmInfo[],
});

export function algorithmInfo(): AlgorithmInfo | undefined {
  return algorithm.catalog.find((a) => a.id === algorithm.algo);
}

// ── Training state ──────────────────────────────────────────────────────
export const training = $state({
  // Session info mirrored from the backend.
  hasSession: false,
  paramCount: 0,
  iteration: 0,
  currentK: 1,
  solveRateByK: {} as Record<string, number>,
  device: '',

  // One record per iteration — the loss chart's data.
  lossHistory: [] as IterationRecord[],

  // Foreground "how many iterations" control + run prefs.
  iterationsPerRun: 200,
  evalEveryN: 100,
  evalN: 80,
  cadenceMinutes: 20,
  // Foreground curriculum: auto-promote k when the current depth is mastered.
  autoAdvance: false,

  // Last solve-rate eval result.
  lastEval: null as { solve_rate: number; mean_solution_len: number | null; k: number } | null,

  // Foreground run-state, lifted to the store so the train loop (a JS closure)
  // and the UI stay in sync across tab unmount/remount.
  busy: false,
  statusMsg: 'idle' as string,
  running: false,
  abort: false,

  // Background run status (mirrored from the backend over /ws + polling).
  run: null as RunStatus | null,

  // Checkpoint UI.
  checkpointFilename: 'cube.pt',
  availableCheckpoints: [] as CheckpointFile[],
  autoLoadOnRestart: false,
});

const MAX_LOSS_HISTORY = 5000;

export function pushIteration(record: IterationRecord) {
  training.lossHistory.push(record);
  if (training.lossHistory.length > MAX_LOSS_HISTORY) {
    training.lossHistory = training.lossHistory.slice(-MAX_LOSS_HISTORY);
  }
  training.iteration = record.iteration;
  training.currentK = record.k;
}

/** Mirror a session summary (+ optional history) from the backend. */
export function applySessionState(s: SessionState) {
  training.hasSession = !!s.has_session;
  if (!s.has_session) {
    training.paramCount = 0;
    training.iteration = 0;
    training.currentK = 1;
    training.solveRateByK = {};
    training.lossHistory = [];
    training.lastEval = null;
    return;
  }
  training.paramCount = s.param_count ?? 0;
  training.iteration = s.iteration ?? 0;
  training.currentK = s.current_k ?? 1;
  training.solveRateByK = s.solve_rate_by_k ?? {};
  training.device = s.device ?? '';
  if (s.loss_history) {
    training.lossHistory = s.loss_history.map((r) => ({ ...r }));
  }
}

// ── Report state (RL Coach written) ─────────────────────────────────────
export const report = $state({
  markdown: '',
  final: false,
  updatedAt: null as number | null,
});

// ── Chat (RL Coach) state — domain-agnostic ─────────────────────────────
export type ChatRole = 'user' | 'assistant';

export type ChatItem =
  | { kind: 'text'; role: ChatRole; text: string; streaming?: boolean }
  | {
      kind: 'tool';
      id: string;
      name: string;
      input: unknown;
      status: 'running' | 'success' | 'error';
      result?: string;
    }
  | { kind: 'system'; text: string }
  | { kind: 'error'; message: string };

export type ChatSessionSummary = {
  session_id: string;
  summary: string | null;
  message_count: number | null;
  last_modified: number | null;
  created_at: number | null;
};

export const CONTEXT_WINDOW = 1_000_000;

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
  visible: false,
  paneWidth: CHAT_PANE_DEFAULT_WIDTH,
  items: [] as ChatItem[],
  activeSessionId: null as string | null,
  sessions: [] as ChatSessionSummary[],
  turn: 'idle' as 'idle' | 'streaming',
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
    total_cost_usd: 0,
  },
  draft: '',
});

const CHAT_PREFS_KEY = 'agentic-cube.chat-prefs.v1';
type ChatPrefs = { visible: boolean; activeSessionId: string | null; paneWidth?: number };

function readChatPrefs(): Partial<ChatPrefs> {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(CHAT_PREFS_KEY);
    return raw ? (JSON.parse(raw) as Partial<ChatPrefs>) : {};
  } catch {
    return {};
  }
}

export function persistChatPrefs() {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(
      CHAT_PREFS_KEY,
      JSON.stringify({
        visible: chat.visible,
        activeSessionId: chat.activeSessionId,
        paneWidth: chat.paneWidth,
      })
    );
  } catch {
    // ignore
  }
}

const _initialChatPrefs = readChatPrefs();
if (typeof _initialChatPrefs.visible === 'boolean') chat.visible = _initialChatPrefs.visible;
if (typeof _initialChatPrefs.activeSessionId === 'string')
  chat.activeSessionId = _initialChatPrefs.activeSessionId;
if (typeof _initialChatPrefs.paneWidth === 'number')
  chat.paneWidth = clampChatPaneWidth(_initialChatPrefs.paneWidth);

// Per-tab client id for state-sync echo prevention.
function _genClientId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
export const CLIENT_ID = _genClientId();

// ── Checkpoint prefs persistence ────────────────────────────────────────
const CKPT_PREFS_KEY = 'agentic-cube.checkpoint-prefs.v1';
type CheckpointPrefs = { filename: string; autoLoadOnRestart: boolean };

function readCheckpointPrefs(): Partial<CheckpointPrefs> {
  if (typeof localStorage === 'undefined') return {};
  try {
    const raw = localStorage.getItem(CKPT_PREFS_KEY);
    return raw ? (JSON.parse(raw) as Partial<CheckpointPrefs>) : {};
  } catch {
    return {};
  }
}

export function persistCheckpointPrefs() {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(
      CKPT_PREFS_KEY,
      JSON.stringify({
        filename: training.checkpointFilename,
        autoLoadOnRestart: training.autoLoadOnRestart,
      })
    );
  } catch {
    // ignore
  }
}

const _initialCkptPrefs = readCheckpointPrefs();
if (typeof _initialCkptPrefs.filename === 'string' && _initialCkptPrefs.filename)
  training.checkpointFilename = _initialCkptPrefs.filename;
if (typeof _initialCkptPrefs.autoLoadOnRestart === 'boolean')
  training.autoLoadOnRestart = _initialCkptPrefs.autoLoadOnRestart;

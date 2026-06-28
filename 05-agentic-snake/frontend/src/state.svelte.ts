import type {
  AlgorithmId,
  AlgorithmInfo,
  CheckpointFile,
  EpisodeRecord,
  ObservationModel,
  SessionState,
} from './api';

export type TabId =
  | 'orientation'
  | 'environment'
  | 'algorithm'
  | 'training'
  | 'watch'
  | 'debrief';

export const ui = $state({
  activeTab: 'orientation' as TabId,
});

// ── UI prefs persistence (active tab survives reloads) ──────────────────
const UI_PREFS_KEY = 'agentic-snake.ui-prefs.v1';
const VALID_TABS: TabId[] = [
  'orientation', 'environment', 'algorithm', 'training', 'watch', 'debrief',
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

// ── Environment state ───────────────────────────────────────────────────
export const environment = $state({
  width: 10,
  height: 10,
  observation: 'features' as ObservationModel,
  reward: {
    food: 1.0,
    death: -1.0,
    step: 0.0,
    toward_food: 0.0,
    away_from_food: 0.0,
  },
});

// ── Algorithm state ─────────────────────────────────────────────────────
export const algorithm = $state({
  loaded: false,
  algo: 'qlearning' as AlgorithmId,
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
  usesNetwork: false,
  paramCount: 0,
  episode: 0,
  bestScore: 0,
  device: '',

  // One record per episode — the score chart's data.
  scoreHistory: [] as EpisodeRecord[],

  // The "how many episodes" control, and greedy-eval cadence.
  episodesPerRun: 200,
  evalEveryN: 25,

  // Last greedy-eval result (mean/best score over N games).
  lastEval: null as { mean_score: number; best_score: number; mean_length: number } | null,

  // Run-state, lifted to the store so the train loop (a JS closure) and the
  // UI stay in sync across tab unmount/remount.
  busy: false,
  statusMsg: 'idle' as string,
  running: false,
  abort: false,

  // Checkpoint UI.
  checkpointFilename: 'snake.pt',
  availableCheckpoints: [] as CheckpointFile[],
  autoSave: false,
  autoLoadOnRestart: false,
});

const MAX_SCORE_HISTORY = 5000;

export function pushEpisode(record: EpisodeRecord) {
  training.scoreHistory.push(record);
  if (training.scoreHistory.length > MAX_SCORE_HISTORY) {
    training.scoreHistory = training.scoreHistory.slice(-MAX_SCORE_HISTORY);
  }
  training.episode = record.episode;
  if (record.score > training.bestScore) training.bestScore = record.score;
}

/** Mirror a session summary (+ optional history) from the backend. */
export function applySessionState(s: SessionState) {
  training.hasSession = !!s.has_session;
  if (!s.has_session) {
    training.paramCount = 0;
    training.episode = 0;
    training.bestScore = 0;
    training.scoreHistory = [];
    training.lastEval = null;
    return;
  }
  training.usesNetwork = !!s.uses_network;
  training.paramCount = s.param_count ?? 0;
  training.episode = s.episode ?? 0;
  training.bestScore = s.best_score ?? 0;
  training.device = s.device ?? '';
  if (s.score_history) {
    training.scoreHistory = s.score_history.map((r) => ({ ...r }));
  }
}

// ── Chat (RL Coach) state — ported from 04, domain-agnostic ─────────────
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

const CHAT_PREFS_KEY = 'agentic-snake.chat-prefs.v1';
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
const CKPT_PREFS_KEY = 'agentic-snake.checkpoint-prefs.v1';
type CheckpointPrefs = { filename: string; autoSave: boolean; autoLoadOnRestart: boolean };

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
        autoSave: training.autoSave,
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
if (typeof _initialCkptPrefs.autoSave === 'boolean')
  training.autoSave = _initialCkptPrefs.autoSave;
if (typeof _initialCkptPrefs.autoLoadOnRestart === 'boolean')
  training.autoLoadOnRestart = _initialCkptPrefs.autoLoadOnRestart;

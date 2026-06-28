// Translates an MCP tool name + its arguments into a short English-language
// phrase suitable for the chat transcript. The full args + result still
// live in the hover popover — this is just the inline label.
//
// Phrasing convention: present-progressive ("Setting…", "Training…") so it
// reads naturally whether the call is in flight, succeeded, or failed.

type ToolInput = Record<string, unknown> | null | undefined;

function s(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null;
}

function n(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function pluralize(count: number, singular: string, plural?: string): string {
  return count === 1 ? singular : (plural ?? singular + 's');
}

export function toolLabel(name: string, input: unknown): string {
  const args = (typeof input === 'object' && input !== null
    ? (input as Record<string, unknown>)
    : {}) as ToolInput;

  switch (name) {
    // ── Read tools ────────────────────────────────────────────────────
    case 'get_pipeline_state':
      return 'Checking pipeline state';
    case 'get_recent_progress': {
      const count = n(args?.n);
      return count
        ? `Reading last ${count} ${pluralize(count, 'episode')}`
        : 'Reading recent progress';
    }
    case 'list_algorithms':
      return 'Reviewing the available algorithms';
    case 'list_checkpoints':
      return 'Listing saved checkpoints';
    case 'list_devices':
      return 'Checking available devices';

    // ── Configuration ─────────────────────────────────────────────────
    case 'set_environment':
      return 'Configuring the environment';
    case 'set_algorithm': {
      const algo = s(args?.algo);
      return algo ? `Switching to ${algo}` : 'Choosing the algorithm';
    }
    case 'set_hyperparameters':
      return 'Setting hyperparameters';

    // ── Training ──────────────────────────────────────────────────────
    case 'init_session':
      return 'Initializing the agent';
    case 'reset_session':
      return 'Resetting the agent';
    case 'train_n_episodes': {
      const count = n(args?.n);
      return count
        ? `Training (${count} ${pluralize(count, 'episode')})`
        : 'Training';
    }
    case 'evaluate': {
      const count = n(args?.n);
      return count ? `Evaluating over ${count} games` : 'Evaluating (greedy)';
    }
    case 'watch_agent_play':
      return 'Watching the agent play a game';

    // ── Checkpoints ───────────────────────────────────────────────────
    case 'save_checkpoint': {
      const fn = s(args?.filename);
      return fn ? `Saving checkpoint "${fn}"` : 'Saving checkpoint';
    }
    case 'load_checkpoint': {
      const fn = s(args?.filename);
      return fn ? `Loading checkpoint "${fn}"` : 'Loading checkpoint';
    }
    case 'delete_checkpoint': {
      const fn = s(args?.filename);
      return fn ? `Deleting checkpoint "${fn}"` : 'Deleting checkpoint';
    }

    // ── Device ────────────────────────────────────────────────────────
    case 'select_device': {
      const dev = s(args?.name);
      return dev ? `Switching to ${dev.toUpperCase()}` : 'Switching device';
    }

    // ── SDK built-ins ─────────────────────────────────────────────────
    // The Claude Agent SDK uses ToolSearch to load deferred tool schemas
    // before invoking them — show a short generic label so it doesn't
    // look like noise in the transcript.
    case 'ToolSearch':
      return 'Looking up tool schema';

    default:
      // Fallback — turn snake_case_thing into "Snake case thing".
      return name
        .replace(/_/g, ' ')
        .replace(/^./, (c) => c.toUpperCase());
  }
}

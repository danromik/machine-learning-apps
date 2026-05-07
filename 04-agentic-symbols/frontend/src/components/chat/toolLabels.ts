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
    case 'get_recent_loss': {
      const count = n(args?.n);
      return count
        ? `Reading last ${count} loss values`
        : 'Reading recent loss values';
    }
    case 'list_symbol_categories':
      return 'Listing symbol categories';
    case 'list_curated_fonts':
      return 'Listing available fonts';
    case 'list_synthesis_presets':
      return 'Reviewing synthesis presets';
    case 'list_checkpoints':
      return 'Listing saved checkpoints';
    case 'list_devices':
      return 'Checking available devices';

    // ── Synthesis ─────────────────────────────────────────────────────
    case 'set_symbol_categories': {
      const cats = Array.isArray(args?.categories) ? args!.categories : null;
      return cats
        ? `Selecting ${cats.length} symbol ${pluralize(cats.length, 'category', 'categories')}`
        : 'Setting symbol categories';
    }
    case 'set_font_usage': {
      const usage = args?.fontUsage as Record<string, string> | undefined;
      if (usage) {
        const train = Object.values(usage).filter((r) => r === 'train').length;
        const val = Object.values(usage).filter((r) => r === 'val').length;
        return `Configuring fonts (${train} train, ${val} val)`;
      }
      return 'Configuring font usage';
    }
    case 'set_augmentation':
      return 'Setting data augmentation';
    case 'apply_synthesis_preset': {
      const name = s(args?.name);
      return name ? `Applying "${name}" preset` : 'Applying synthesis preset';
    }

    // ── Architecture ──────────────────────────────────────────────────
    case 'set_architecture': {
      const layers = Array.isArray(args?.layers) ? args!.layers : null;
      return layers
        ? `Setting model architecture (${layers.length} ${pluralize(layers.length, 'layer')})`
        : 'Setting model architecture';
    }
    case 'set_hyperparameters':
      return 'Setting hyperparameters';

    // ── Training ──────────────────────────────────────────────────────
    case 'init_session':
      return 'Initializing training session';
    case 'reset_session':
      return 'Resetting training session';
    case 'train_n_batches': {
      const count = n(args?.n);
      return count
        ? `Training (${count} ${pluralize(count, 'batch', 'batches')})`
        : 'Training';
    }
    case 'eval_on_val': {
      const count = n(args?.count);
      return count
        ? `Evaluating on ${count} validation samples`
        : 'Evaluating on validation';
    }

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

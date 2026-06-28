<script lang="ts">
  import { training, report, chat, persistChatPrefs } from '../../state.svelte';
  import { api, streamChat } from '../../api';
  import { applyChatEvent } from '../chat/chatReducer';
  import Markdown from '../Markdown.svelte';

  let finishing = $state(false);

  let run = $derived(training.run);
  let solveRows = $derived(
    Object.entries(training.solveRateByK)
      .map(([k, v]) => ({ k: Number(k), rate: v }))
      .sort((a, b) => a.k - b.k)
  );

  const FINISH_PROMPT =
    'Training is finished — the user clicked "Finish Training". If useful, run a ' +
    'final evaluation, then call generate_final_report with an honest, well-' +
    'structured debrief: solve-rate by scramble depth, what was achieved (is the ' +
    'cube genuinely solvable now?), the key decisions and what they taught, the ' +
    'limits hit, and real next steps.';

  async function finish() {
    if (finishing) return;
    finishing = true;
    try {
      if (training.run?.running) {
        try {
          training.run = await api.runStop();
        } catch {
          /* ignore */
        }
      }
      // Drive the RL Coach to write the final report. The resulting
      // report_final WS event navigates the app to the Debrief tab.
      chat.visible = true;
      persistChatPrefs();
      if (chat.turn !== 'streaming') {
        chat.turn = 'streaming';
        try {
          for await (const ev of streamChat(FINISH_PROMPT, chat.activeSessionId)) {
            applyChatEvent(ev);
          }
        } catch (e) {
          chat.items.push({ kind: 'error', message: `finish failed: ${e}` });
        } finally {
          chat.turn = 'idle';
        }
      }
    } finally {
      finishing = false;
    }
  }

  function fmtTime(ts: number | null): string {
    if (!ts) return '';
    try {
      return new Date(ts * 1000).toLocaleTimeString();
    } catch {
      return '';
    }
  }
</script>

<div class="h-full overflow-auto px-6 py-6">
  <div class="max-w-4xl mx-auto flex flex-col gap-5">
    <div class="flex items-start justify-between gap-4">
      <div>
        <h2 class="text-lg font-bold text-[var(--color-heading)]">Progress Report</h2>
        <p class="text-sm text-[var(--color-muted)] mt-1">
          A live report the RL Coach maintains as training proceeds.
          {#if report.updatedAt}· updated {fmtTime(report.updatedAt)}{/if}
        </p>
      </div>
      <button type="button" class="btn-primary shrink-0" disabled={finishing} onclick={finish}>
        {finishing ? 'Finishing…' : 'Finish Training'}
      </button>
    </div>

    <!-- Status header -->
    <div class="grid gap-3" style="grid-template-columns: repeat(4, 1fr)">
      {#each [{ label: 'Run', val: run && run.state !== 'idle' ? run.state : '—' }, { label: 'Iterations', val: String(training.iteration) }, { label: 'Curriculum k', val: String(training.currentK) }, { label: 'Checkpoint', val: run?.last_checkpoint ?? '—' }] as stat}
        <div class="card p-3 min-w-0">
          <div class="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
            {stat.label}
          </div>
          <div class="text-sm font-bold text-[var(--color-heading)] font-mono truncate">
            {stat.val}
          </div>
        </div>
      {/each}
    </div>

    {#if solveRows.length}
      <div class="card p-4">
        <div class="text-xs font-semibold text-[var(--color-heading)] mb-2">Solve-rate by depth</div>
        <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono text-[var(--color-text)]">
          {#each solveRows as row}
            <span>k={row.k}: {(row.rate * 100).toFixed(0)}%</span>
          {/each}
        </div>
      </div>
    {/if}

    <!-- The report itself -->
    <div class="card p-5">
      {#if report.markdown}
        <Markdown source={report.markdown} />
      {:else}
        <div class="text-sm text-[var(--color-muted)]">
          No report yet. Start a background run on the Training tab and ask the RL
          Coach to keep this updated — it will write a summary here as the run
          progresses, then a full debrief when you click Finish Training.
        </div>
      {/if}
    </div>
  </div>
</div>

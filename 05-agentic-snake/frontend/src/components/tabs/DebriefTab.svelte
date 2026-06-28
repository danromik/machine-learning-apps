<script lang="ts">
  import { ui, environment, algorithm, algorithmInfo, training, chat, persistChatPrefs } from '../../state.svelte';

  // "Great play" threshold scales with grid difficulty — bigger boards are
  // harder, so we expect proportionally fewer apples to count as "doing well".
  let goodThreshold = $derived(Math.max(6, Math.round((environment.width * environment.height) / 12)));

  let recentMean = $derived.by(() => {
    const recent = training.scoreHistory.slice(-50);
    if (!recent.length) return 0;
    return recent.reduce((a, r) => a + r.score, 0) / recent.length;
  });

  type Bucket =
    | 'no-agent'
    | 'untrained'
    | 'early'
    | 'progressing'
    | 'strong';

  let bucket = $derived.by((): Bucket => {
    if (!training.hasSession) return 'no-agent';
    if (training.episode < 5) return 'untrained';
    if (recentMean < 1) return 'early';
    if (recentMean < goodThreshold) return 'progressing';
    return 'strong';
  });

  const MESSAGES: Record<Bucket, string> = {
    'no-agent':
      'No agent exists yet. Head to the Training tab and initialize one, then train it for a few hundred episodes.',
    untrained:
      'The agent is initialized but has barely played. Train it for a few hundred episodes and watch the score chart — early on it mostly explores at random.',
    early:
      "The agent is still scoring near zero — it's learning to survive before it learns to eat. Keep training; with the food-direction features it should start chasing apples soon.",
    progressing:
      "It's learning! The score is climbing as exploration (ε) decays and the agent trusts its policy more. Train further, or try a greedy Evaluate to see its real ability.",
    strong:
      'Strong play! The agent reliably chases food and avoids trapping itself. Try a harder grid, a different algorithm, or watch a game to see its strategy.',
  };

  function goTrain() {
    ui.activeTab = 'training';
  }
  function askCoach() {
    chat.visible = true;
    persistChatPrefs();
  }
</script>

<div class="h-full overflow-auto px-6 py-8">
  <div class="max-w-2xl mx-auto flex flex-col gap-6">
    <h2 class="text-xl font-bold text-[var(--color-heading)]">Debrief</h2>

    <div class="card p-5 border-l-2 border-l-[var(--color-accent)] text-[var(--color-text)] leading-relaxed">
      {MESSAGES[bucket]}
    </div>

    <div class="grid gap-3" style="grid-template-columns: 1fr 1fr">
      <div class="card p-4">
        <div class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide mb-2">
          Setup
        </div>
        <dl class="text-sm flex flex-col gap-1">
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Algorithm</dt>
            <dd>{algorithmInfo()?.label ?? '—'}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Grid</dt>
            <dd class="font-mono">{environment.width}×{environment.height}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Observation</dt>
            <dd>{environment.observation === 'grid' ? 'full grid (CNN)' : '11 features'}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Learning rate</dt>
            <dd class="font-mono">{algorithm.hyperparameters.lr ?? '—'}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Discount γ</dt>
            <dd class="font-mono">{algorithm.hyperparameters.gamma ?? '—'}</dd>
          </div>
        </dl>
      </div>

      <div class="card p-4">
        <div class="text-xs font-semibold text-[var(--color-heading)] uppercase tracking-wide mb-2">
          Progress
        </div>
        <dl class="text-sm flex flex-col gap-1">
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Episodes trained</dt>
            <dd class="font-mono">{training.episode}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Best score</dt>
            <dd class="font-mono">{training.bestScore}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Recent avg (50)</dt>
            <dd class="font-mono">{recentMean.toFixed(2)}</dd>
          </div>
          <div class="flex justify-between">
            <dt class="text-[var(--color-muted)]">Greedy eval</dt>
            <dd class="font-mono">
              {training.lastEval ? training.lastEval.mean_score.toFixed(2) : '—'}
            </dd>
          </div>
        </dl>
      </div>
    </div>

    <div class="flex gap-3">
      <button type="button" class="btn-primary" onclick={goTrain}>Go to training →</button>
      <button type="button" class="btn-outline" onclick={askCoach}>Ask the RL Coach</button>
    </div>
  </div>
</div>

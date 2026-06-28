<script lang="ts">
  import { ui, chat, persistChatPrefs } from '../../state.svelte';

  const STEPS: { title: string; body: string }[] = [
    {
      title: 'Environment',
      body: 'Set the grid size and the reward the agent gets for eating, dying, and moving. Reward shaping is the single biggest lever on behavior.',
    },
    {
      title: 'Algorithm',
      body: 'Pick how the agent learns: tabular Q-learning (no network), DQN (deep value-based), or REINFORCE (policy gradient) — and tune its hyperparameters.',
    },
    {
      title: 'Training',
      body: 'Initialize the agent and train it for as many episodes (full games) as you like. Watch the score climb live on the progress chart.',
    },
    {
      title: 'Watch',
      body: 'Watch the trained agent play, frame by frame, with an overlay of what it is thinking (Q-values or action probabilities) at each step.',
    },
  ];
</script>

<div class="h-full overflow-auto">
  <div class="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-6">
    <h1 class="text-2xl font-bold text-[var(--color-heading)] tracking-tight">
      Welcome to Agentic Snake Trainer!
    </h1>

    <p class="text-base text-[var(--color-text)] leading-relaxed">
      Here we train a <strong>reinforcement-learning</strong> agent to play Snake
      on a grid. Unlike the other apps in this suite, there is no dataset — the
      agent learns purely from the <em>reward</em> it earns by playing, getting
      better the more games it plays. The pipeline:
    </p>

    <ol class="flex flex-col gap-3">
      {#each STEPS as step, i}
        <li class="flex items-start gap-3">
          <span
            class="inline-flex items-center justify-center w-7 h-7 rounded-full
                   bg-[var(--color-accent)] text-[var(--color-on-accent)]
                   font-mono text-sm font-semibold shrink-0 mt-0.5"
            aria-hidden="true"
          >
            {i + 1}
          </span>
          <div class="flex-1 text-[var(--color-text)] leading-relaxed">
            <span class="font-semibold text-[var(--color-heading)]">{step.title}</span>
            <span class="text-[var(--color-muted)]"> — {step.body}</span>
          </div>
        </li>
      {/each}
    </ol>

    <div
      class="card p-4 text-sm text-[var(--color-text)] leading-relaxed
             border-l-2 border-l-[var(--color-accent)]"
    >
      <span class="font-semibold text-[var(--color-heading)]">Meet the RL Coach.</span>
      You don't have to drive any of this yourself. Open the chat and ask the
      embedded RL Coach to set things up, run training, and explain what's
      happening — it drives the exact same pipeline you do, and you'll see every
      move it makes reflected live in these tabs.
    </div>

    <div class="flex items-center gap-3 pt-2 flex-wrap">
      <button type="button" class="btn-primary" onclick={() => (ui.activeTab = 'environment')}>
        Configure the environment →
      </button>
      <button
        type="button"
        class="btn-outline"
        onclick={() => {
          chat.visible = true;
          persistChatPrefs();
        }}
      >
        Ask the RL Coach to do it
      </button>
    </div>
  </div>
</div>

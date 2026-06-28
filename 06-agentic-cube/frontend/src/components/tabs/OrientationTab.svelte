<script lang="ts">
  import { ui, chat, persistChatPrefs } from '../../state.svelte';

  const STEPS = [
    {
      title: 'Cube',
      body: 'Pick a 2×2×2 (fully solvable, fast) or 3×3×3 (the real challenge), and set the reverse-scramble curriculum — how deep to scramble and when to go deeper.',
    },
    {
      title: 'Algorithm',
      body: 'Value iteration learns a cost-to-go heuristic using the cube’s known model — no sparse-reward exploration needed. Tune the network and learning rate.',
    },
    {
      title: 'Training',
      body: 'Train in short foreground bursts, or launch a long background run that ramps the curriculum and checkpoints itself — overnight-capable.',
    },
    {
      title: 'Watch',
      body: 'Watch the learned solver take a scrambled cube back to solved in real 3D — drag to rotate, scroll to zoom.',
    },
    {
      title: 'Progress Report & Debrief',
      body: 'The RL Coach keeps a live training report as the run proceeds; click Finish Training for a final written debrief.',
    },
  ];

  function openCoach() {
    chat.visible = true;
    persistChatPrefs();
  }
</script>

<div class="h-full overflow-auto px-6 py-8">
  <div class="max-w-3xl mx-auto">
    <h1 class="text-2xl font-bold text-[var(--color-heading)]">Agentic Cube Trainer</h1>
    <p class="mt-3 text-[var(--color-text)] leading-relaxed">
      Teach a neural network to solve a Rubik's Cube — and meet the
      <strong>sparse-reward problem</strong> head on. A scrambled 3×3 cube has one
      solved state among ~4.3×10¹⁹, so the trial-and-error RL from the Snake app
      would never stumble onto the goal. The fix: exploit the cube's
      <em>known model</em> with <strong>value iteration</strong> over a learned
      cost-to-go function, trained on a <strong>reverse-scramble curriculum</strong>
      (scramble <code>k</code> moves from solved, ramp <code>k</code> up), and paired
      with a beam search at solve time.
    </p>
    <p class="mt-3 text-[var(--color-text)] leading-relaxed">
      The <strong>RL Coach</strong> can drive this whole pipeline for you — including
      launching a long overnight training run it checks in on periodically, writing a
      live report as it goes. Or drive it yourself through the tabs.
    </p>

    <ol class="mt-6 flex flex-col gap-3">
      {#each STEPS as step, i}
        <li class="card p-4 flex gap-3">
          <span
            class="shrink-0 w-7 h-7 rounded-full bg-[var(--color-accent)] text-white
                   font-bold flex items-center justify-center text-sm">{i + 1}</span
          >
          <div>
            <div class="font-semibold text-[var(--color-heading)]">{step.title}</div>
            <div class="text-sm text-[var(--color-muted)] mt-0.5">{step.body}</div>
          </div>
        </li>
      {/each}
    </ol>

    <div class="mt-6 flex gap-3">
      <button type="button" class="btn-primary" onclick={() => (ui.activeTab = 'cube')}>
        Start with the Cube →
      </button>
      <button type="button" class="btn-outline" onclick={openCoach}>Ask the RL Coach</button>
    </div>
  </div>
</div>

<script lang="ts">
  import { ui } from '../../state.svelte';

  const STEPS: { title: string; body: string }[] = [
    {
      title: 'Data Synthesis',
      body: 'Choose which symbols to classify, which fonts to render them in, and what augmentations (noise, skew) to apply during training.',
    },
    {
      title: 'Model Architecture',
      body: 'Design the neural network layer-by-layer (Conv2D, MaxPool, Linear, …) and pick hyperparameters like learning rate, batch size, and optimizer.',
    },
    {
      title: 'Training',
      body: 'Run training and watch loss and validation curves update live as the model learns. Save checkpoints, pause, or restart at any time.',
    },
    {
      title: 'Inference',
      body: 'Type text and watch the trained model classify each glyph in real time, with top-K predictions and confidences.',
    },
  ];

  function startConfiguring() {
    ui.activeTab = 'data';
  }
</script>

<div class="h-full overflow-auto">
  <div class="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-6">
    <h1 class="text-2xl font-bold text-[var(--color-heading)] tracking-tight">
      Welcome to Agentic Symbol Trainer!
    </h1>

    <p class="text-base text-[var(--color-text)] leading-relaxed">
      In this app we will design and train a model that classifies text symbols
      in mathematical writing. The training pipeline consists of the following
      steps:
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

    <div class="flex items-center gap-3 pt-4 flex-wrap">
      <span class="text-base text-[var(--color-text)]">Ready to start?</span>
      <button type="button" class="btn-primary" onclick={startConfiguring}>
        Configure the data synthesis →
      </button>
    </div>
  </div>
</div>

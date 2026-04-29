<script lang="ts">
  import { ui } from '../../state.svelte';

  const STEPS: { title: string; body: string }[] = [
    {
      title: 'Data Acquisition',
      body: 'Download Imagenette — fast.ai\'s 10-class subset of ImageNet — and choose what augmentations to apply during training.',
    },
    {
      title: 'Model Architecture',
      body: 'Pick one of three preset architectures (LeNet-5, AlexNet, ResNet-18) or design your own layer-by-layer with the drag-and-drop canvas.',
    },
    {
      title: 'Training',
      body: 'Run training and watch loss and validation curves update live. Save checkpoints, pause, or restart at any time.',
    },
    {
      title: 'Inference',
      body: 'Upload an image (or sample one from the validation set) and see how your model classifies it, with top-K predictions and confidences.',
    },
  ];

  function startConfiguring() {
    ui.activeTab = 'data';
  }
</script>

<div class="h-full overflow-auto">
  <div class="max-w-2xl mx-auto px-6 py-10 flex flex-col gap-6">
    <h1 class="text-2xl font-bold text-[var(--color-heading)] tracking-tight">
      Welcome to the Image Classifier!
    </h1>

    <p class="text-base text-[var(--color-text)] leading-relaxed">
      In this app you'll design and train a neural network that classifies
      photographs of everyday objects. We'll use Imagenette — fast.ai's
      10-class subset of ImageNet, the dataset AlexNet famously trained on —
      and walk you through three landmark architectures from the history of
      deep learning. The pipeline:
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

    <div class="card p-4 flex flex-col gap-2 text-sm text-[var(--color-text)] leading-relaxed">
      <h2 class="text-sm font-semibold text-[var(--color-heading)]">
        Three architectures, one decade of history
      </h2>
      <p>
        <span class="font-semibold">LeNet-5</span> (1998) is the classic
        pre-AlexNet CNN — five layers, designed for handwritten digits.
        <span class="font-semibold">AlexNet</span> (2012) is the architecture
        that won ILSVRC and kicked off modern deep learning. <span class="font-semibold">ResNet-18</span>
        (2015) is a more modern design with residual connections that allow
        much deeper networks to train successfully. You can also build your
        own from scratch.
      </p>
    </div>

    <div class="flex items-center gap-3 pt-4 flex-wrap">
      <span class="text-base text-[var(--color-text)]">Ready to start?</span>
      <button type="button" class="btn-primary" onclick={startConfiguring}>
        Acquire the dataset →
      </button>
    </div>
  </div>
</div>

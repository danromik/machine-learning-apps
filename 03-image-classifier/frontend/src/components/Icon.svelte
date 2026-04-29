<!-- Inline SVG icon — paths sourced from Lucide (MIT). -->
<script lang="ts">
  type Name =
    | 'play'
    | 'stop'
    | 'step'
    | 'refresh'
    | 'restart'
    | 'download'
    | 'save'
    | 'x'
    | 'edit'
    | 'brain'
    | 'cpu'
    | 'loader'
    | 'check'
    | 'sigma'
    | 'layers'
    | 'gauge'
    | 'type'
    | 'chevron-right';

  let { name, size = 16, strokeWidth = 2, class: klass = '' }:
    { name: Name; size?: number; strokeWidth?: number; class?: string } = $props();

  const paths: Record<Name, string> = {
    play: '<polygon points="6 3 20 12 6 21 6 3"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1"/>',
    step: '<polygon points="5 4 15 12 5 20 5 4"/><line x1="19" x2="19" y1="5" y2="19"/>',
    refresh:
      '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    restart:
      '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    download:
      '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    save:
      '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    edit:
      '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/>',
    brain:
      '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44A2.5 2.5 0 0 1 3 17.5a2.5 2.5 0 0 1-1.32-4.24A2.5 2.5 0 0 1 2 10a2.5 2.5 0 0 1 1.98-2.44A2.5 2.5 0 0 1 5 4.5 2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44A2.5 2.5 0 0 0 21 17.5a2.5 2.5 0 0 0 1.32-4.24A2.5 2.5 0 0 0 22 10a2.5 2.5 0 0 0-1.98-2.44A2.5 2.5 0 0 0 19 4.5 2.5 2.5 0 0 0 14.5 2z"/>',
    cpu:
      '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    loader:
      '<path d="M12 2v4"/><path d="m16.2 7.8 2.9-2.9"/><path d="M18 12h4"/><path d="m16.2 16.2 2.9 2.9"/><path d="M12 18v4"/><path d="m4.9 19.1 2.9-2.9"/><path d="M2 12h4"/><path d="m4.9 4.9 2.9 2.9"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
    sigma:
      '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l5.4 7.2-5.4 7.2a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2"/>',
    layers:
      '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.91a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    gauge:
      '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    type:
      '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>',
    'chevron-right': '<polyline points="9 18 15 12 9 6"/>',
  };
</script>

<svg
  xmlns="http://www.w3.org/2000/svg"
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width={strokeWidth}
  stroke-linecap="round"
  stroke-linejoin="round"
  class={klass}
>
  {@html paths[name]}
</svg>

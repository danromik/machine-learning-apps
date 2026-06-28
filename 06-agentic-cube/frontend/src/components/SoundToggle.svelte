<script lang="ts">
  import Icon from './Icon.svelte';
  import { sound } from '../state.svelte';
  import { setSoundEnabled } from '../sound';

  function toggle() {
    sound.enabled = !sound.enabled;
    setSoundEnabled(sound.enabled); // persists + resumes the audio context
  }
</script>

<button
  type="button"
  onclick={toggle}
  title={sound.enabled ? 'Sound on — click to mute' : 'Sound off — click to unmute'}
  class="flex items-center gap-1.5 text-xs text-[var(--color-muted)] hover:text-[var(--color-text)]
         transition-colors"
>
  <Icon name={sound.enabled ? 'volume' : 'volume-x'} size={14} />
  <span>Sound</span>
</button>

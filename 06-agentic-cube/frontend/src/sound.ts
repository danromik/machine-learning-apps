// Synthesized sound effects for the 3D cube playback (Web Audio API — no
// bundled audio assets). A single master on/off, persisted to localStorage and
// toggled from the status bar. All effects are no-ops when disabled or when
// Web Audio is unavailable.
//
//   playSolve() — bright ascending major arpeggio when a solved cube finishes
//   playTurn()  — a short percussive clack per layer turn
//   playFail()  — a soft descending two-note when an attempt ends unsolved
//
// resumeAudio() must be called from within a user gesture (the Play click) so
// the AudioContext is allowed to start under browser autoplay policy.

const KEY = 'agentic-cube.sound.v1';

function readEnabled(): boolean {
  if (typeof localStorage === 'undefined') return true;
  try {
    const v = localStorage.getItem(KEY);
    return v === null ? true : v === '1';
  } catch {
    return true;
  }
}

let enabled = readEnabled();

export function isSoundEnabled(): boolean {
  return enabled;
}

export function setSoundEnabled(v: boolean): void {
  enabled = v;
  try {
    localStorage.setItem(KEY, v ? '1' : '0');
  } catch {
    // ignore quota / private-mode errors
  }
  if (v) resumeAudio();
}

let ctx: AudioContext | null = null;

function audio(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!ctx) {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return null;
    try {
      ctx = new AC();
    } catch {
      return null;
    }
  }
  return ctx;
}

export function resumeAudio(): void {
  const c = audio();
  if (c && c.state === 'suspended') c.resume().catch(() => {});
}

// A single enveloped oscillator note.
function tone(
  c: AudioContext,
  freq: number,
  startOffset: number,
  duration: number,
  type: OscillatorType = 'triangle',
  peak = 0.2
): void {
  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = type;
  osc.frequency.value = freq;
  osc.connect(gain);
  gain.connect(c.destination);
  const t0 = c.currentTime + startOffset;
  gain.gain.setValueAtTime(0.0001, t0);
  gain.gain.exponentialRampToValueAtTime(peak, t0 + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
  osc.start(t0);
  osc.stop(t0 + duration + 0.03);
}

// Short decaying band-passed noise burst — a cube "clack".
function clack(c: AudioContext, peak = 0.07): void {
  const dur = 0.06;
  const len = Math.floor(c.sampleRate * dur);
  const buf = c.createBuffer(1, len, c.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len);
  const src = c.createBufferSource();
  src.buffer = buf;
  const bp = c.createBiquadFilter();
  bp.type = 'bandpass';
  bp.frequency.value = 2200;
  bp.Q.value = 0.7;
  const gain = c.createGain();
  gain.gain.value = peak;
  src.connect(bp);
  bp.connect(gain);
  gain.connect(c.destination);
  src.start();
}

// C5 E5 G5 C6 — a bright "ta-da".
const SOLVE_NOTES = [523.25, 659.25, 783.99, 1046.5];

export function playSolve(): void {
  if (!enabled) return;
  const c = audio();
  if (!c) return;
  resumeAudio();
  SOLVE_NOTES.forEach((f, i) => tone(c, f, i * 0.09, 0.42, 'triangle', 0.22));
}

export function playTurn(): void {
  if (!enabled) return;
  const c = audio();
  if (!c) return;
  clack(c);
}

// G4 → C4, soft and a touch downcast.
export function playFail(): void {
  if (!enabled) return;
  const c = audio();
  if (!c) return;
  resumeAudio();
  tone(c, 392.0, 0, 0.22, 'triangle', 0.16);
  tone(c, 261.63, 0.16, 0.32, 'triangle', 0.16);
}

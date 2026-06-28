// Cube color styles for the 3D views. Each style assigns the six face colors
// (order: U D R L F B) and whether the cube renders solid or as semi-
// transparent "glass". Selected via the Cube Style selector in the status bar
// and persisted to localStorage; consumed by CubeView3D.

export type CubeStyle = {
  id: string;
  label: string;
  colors: number[]; // six 0xRRGGBB values, in face order U D R L F B
  transparent: boolean;
  opacity: number;
};

// Western Rubik's scheme (the original).
const STANDARD = [0xf8f8f8, 0xffd500, 0xc41e3a, 0xff5800, 0x009e60, 0x0051ba];
// Deep gem tones — looks like stained glass in the transparent variant.
const JEWEL = [0x8b5cf6, 0x10b981, 0xe11d48, 0xf59e0b, 0x2563eb, 0xec4899];
// Bright cyber / synthwave palette.
const NEON = [0x22d3ee, 0xff2ec4, 0xff5c39, 0xa3e635, 0xa855f7, 0xfde047];

const GLASS_OPACITY = 0.7;

export const CUBE_STYLES: CubeStyle[] = [
  { id: 'standard', label: 'Standard', colors: STANDARD, transparent: false, opacity: 1 },
  { id: 'jewel', label: 'Jewel', colors: JEWEL, transparent: false, opacity: 1 },
  { id: 'neon', label: 'Neon', colors: NEON, transparent: false, opacity: 1 },
  { id: 'standard-glass', label: 'Standard (Glass)', colors: STANDARD, transparent: true, opacity: GLASS_OPACITY },
  { id: 'jewel-glass', label: 'Jewel (Glass)', colors: JEWEL, transparent: true, opacity: GLASS_OPACITY },
  { id: 'neon-glass', label: 'Neon (Glass)', colors: NEON, transparent: true, opacity: GLASS_OPACITY },
];

export const DEFAULT_CUBE_STYLE = 'standard';

export function cubeStyleById(id: string): CubeStyle {
  return CUBE_STYLES.find((s) => s.id === id) ?? CUBE_STYLES[0];
}

const KEY = 'agentic-cube.cube-style.v1';

export function readCubeStyleId(): string {
  if (typeof localStorage === 'undefined') return DEFAULT_CUBE_STYLE;
  try {
    const v = localStorage.getItem(KEY);
    return v && CUBE_STYLES.some((s) => s.id === v) ? v : DEFAULT_CUBE_STYLE;
  } catch {
    return DEFAULT_CUBE_STYLE;
  }
}

export function persistCubeStyleId(id: string): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(KEY, id);
  } catch {
    // ignore quota / private-mode errors
  }
}

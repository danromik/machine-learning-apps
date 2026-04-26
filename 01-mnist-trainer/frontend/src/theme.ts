export type ThemeName = 'github-light' | 'one-dark';

export const THEMES: { id: ThemeName; label: string }[] = [
  { id: 'github-light', label: 'GitHub Light' },
  { id: 'one-dark', label: 'One Dark' },
];

const STORAGE_KEY = 'mnist-theme';
const DEFAULT: ThemeName = 'github-light';

export function initTheme(): ThemeName {
  const saved = (typeof localStorage !== 'undefined' && localStorage.getItem(STORAGE_KEY)) as
    | ThemeName
    | null;
  const id = saved && THEMES.some((t) => t.id === saved) ? saved : DEFAULT;
  applyTheme(id);
  return id;
}

export function applyTheme(id: ThemeName) {
  document.documentElement.setAttribute('data-theme', id);
  try {
    localStorage.setItem(STORAGE_KEY, id);
  } catch {
    /* ignore */
  }
}

/** Read a CSS custom property from the document root. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

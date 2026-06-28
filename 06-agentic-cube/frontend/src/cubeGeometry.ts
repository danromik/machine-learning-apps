// Client-side construction of a SOLVED cube frame, for static 3D previews
// (the Cube tab, and the Watch tab before a solve is loaded). For a solved
// cube every sticker's color is just the face its outward normal points to, so
// this needs no backend round-trip. Live/scrambled frames always come from the
// backend's render_dict.

import type { CubeFrame } from './api';

// outward normal (as "x,y,z") -> face index / color (U D R L F B)
const NORMAL_TO_FACE: Record<string, number> = {
  '0,1,0': 0, // U
  '0,-1,0': 1, // D
  '1,0,0': 2, // R
  '-1,0,0': 3, // L
  '0,0,1': 4, // F
  '0,0,-1': 5, // B
};

function axisVals(size: number): number[] {
  // size 2 -> [-1, 1]; size 3 -> [-2, 0, 2]
  const out: number[] = [];
  for (let v = -(size - 1); v < size; v += 2) out.push(v);
  return out;
}

export function solvedFrame(size: number): CubeFrame {
  const vals = axisVals(size);
  const ext = size - 1; // extreme coordinate magnitude
  const cubies: CubeFrame['cubies'] = [];
  for (const x of vals) {
    for (const y of vals) {
      for (const z of vals) {
        // Surface cubies only (at least one coord at an extreme).
        if (Math.abs(x) !== ext && Math.abs(y) !== ext && Math.abs(z) !== ext) continue;
        const stickers: CubeFrame['cubies'][number]['stickers'] = [];
        const add = (n: [number, number, number]) => {
          stickers.push({ normal: n, color: NORMAL_TO_FACE[n.join(',')] });
        };
        if (x === ext) add([1, 0, 0]);
        if (x === -ext) add([-1, 0, 0]);
        if (y === ext) add([0, 1, 0]);
        if (y === -ext) add([0, -1, 0]);
        if (z === ext) add([0, 0, 1]);
        if (z === -ext) add([0, 0, -1]);
        cubies.push({ pos: [x, y, z], stickers });
      }
    }
  }
  return { size, coords: vals, cubies, solved: true };
}

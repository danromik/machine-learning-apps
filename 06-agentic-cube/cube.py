"""The Rubik's Cube environment — the *world* the RL Coach learns to solve.

Unlike Snake (`05`), there is no reward-shaped episodic game here. The cube is a
**deterministic, reversible, fully-known** environment: from any state, applying
a move lands you in an exactly-known next state, and every move has an inverse.
That structure is what makes the lesson of this app possible — instead of
learning from sparse trial-and-error reward (which never stumbles onto the one
solved state among ~4.3x10^19), we exploit the known model with **value
iteration over a learned cost-to-go function** (DeepCubeA-style), trained on a
**reverse-scramble curriculum**: scramble the solved cube `k` moves, learn to
predict how many moves away from solved each state is, and ramp `k` up.

Representation
--------------
A cube state is a flat ``int8`` array of **facelet colors**, length ``6*size*size``
(``size`` ∈ {2, 3}). Facelet ``i`` belongs to face ``i // (size*size)``; the six
faces are U, D, R, L, F, B with fixed colors 0..5. Moves are precomputed
**permutations** of that array (built once from an exact cubie-geometry model),
so applying a move is just ``state[perm]`` — fast enough for the millions of
applies value iteration needs.

The geometry (which cubie sits where, which sticker faces which way) is derived
from a right-hand-rule rotation model so the move permutations are provably valid
(``move**4 == identity``, ``move · move' == identity``, scrambles reverse exactly
— see the self-test in ``__main__``). `render_dict` emits per-cubie sticker data
for the three.js 3D Watch view, and each move carries its ``(axis, sign, dir)``
so the frontend can animate the correct layer turning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# ── Axes & faces ────────────────────────────────────────────────────────────
X, Y, Z = 0, 1, 2

# face index -> outward normal (axis, sign); the face's solved color == its index.
#   0 U(+y)  1 D(-y)  2 R(+x)  3 L(-x)  4 F(+z)  5 B(-z)
FACE_NORMALS: dict[int, tuple[int, int]] = {
    0: (Y, +1), 1: (Y, -1), 2: (X, +1), 3: (X, -1), 4: (Z, +1), 5: (Z, -1),
}
NORMAL_TO_FACE: dict[tuple[int, int], int] = {n: f for f, n in FACE_NORMALS.items()}
FACE_NAMES = {0: "U", 1: "D", 2: "R", 3: "L", 4: "F", 5: "B"}
NUM_FACES = 6
NUM_COLORS = 6

# Per-cube-size move catalogs. Each move = (name, axis, sign, dir):
#   axis/sign select the outer layer (coord along `axis` == sign*(size-1));
#   dir = +1/-1 is the right-hand-rule rotation sense about `axis`.
# A 2x2 has no centers, so fixing the D/L/B layers, the {U,R,F} faces (+ inverses)
# generate the whole group — 6 moves. A 3x3 uses all 12 quarter-turn face moves.
_MOVES_2 = [
    ("U", Y, +1, +1), ("U'", Y, +1, -1),
    ("R", X, +1, +1), ("R'", X, +1, -1),
    ("F", Z, +1, +1), ("F'", Z, +1, -1),
]
_MOVES_3 = _MOVES_2 + [
    ("D", Y, -1, +1), ("D'", Y, -1, -1),
    ("L", X, -1, +1), ("L'", X, -1, -1),
    ("B", Z, -1, +1), ("B'", Z, -1, -1),
]


def moves_for(size: int) -> list[tuple[str, int, int, int]]:
    return _MOVES_2 if size == 2 else _MOVES_3


@dataclass
class CubeConfig:
    size: int = 3  # 2 (Pocket Cube) or 3 (standard)

    def __post_init__(self):
        if self.size not in (2, 3):
            raise ValueError("cube size must be 2 or 3")


# ── Geometry helpers ────────────────────────────────────────────────────────
def axis_vals(size: int) -> tuple[int, ...]:
    """Centered integer coordinates per axis: size 2 -> (-1, 1); size 3 -> (-2, 0, 2)."""
    return tuple(range(-(size - 1), size, 2))


def _rot(vec: tuple[int, int, int], axis: int, direction: int) -> tuple[int, int, int]:
    """Rotate `vec` by +/-90 degrees about `axis` (right-hand rule for dir=+1)."""
    x, y, z = vec
    if direction == +1:
        if axis == X:
            return (x, -z, y)
        if axis == Y:
            return (z, y, -x)
        return (-y, x, z)
    # dir == -1 (inverse of the above)
    if axis == X:
        return (x, z, -y)
    if axis == Y:
        return (-z, y, x)
    return (y, -x, z)


def _face_axes(axis: int) -> tuple[int, int]:
    """The two in-face axes (row, col) for a face whose normal is `axis`."""
    others = [a for a in (X, Y, Z) if a != axis]
    return others[0], others[1]


def _facelet_to_pos_normal(size: int, f: int, r: int, c: int):
    """Facelet (face, row, col) -> (cubie position, outward normal)."""
    axis, sign = FACE_NORMALS[f]
    ra, ca = _face_axes(axis)
    vals = axis_vals(size)
    pos = [0, 0, 0]
    pos[axis] = sign * (size - 1)
    pos[ra] = vals[r]
    pos[ca] = vals[c]
    return (pos[0], pos[1], pos[2]), (axis, sign)


def _pos_normal_to_facelet(size: int, pos: tuple[int, int, int], normal: tuple[int, int]) -> int:
    axis, sign = normal
    ra, ca = _face_axes(axis)
    vals = axis_vals(size)
    f = NORMAL_TO_FACE[normal]
    r = vals.index(pos[ra])
    c = vals.index(pos[ca])
    return f * size * size + r * size + c


def _all_facelets(size: int):
    for f in range(NUM_FACES):
        for r in range(size):
            for c in range(size):
                yield f, r, c


def _build_perm(size: int, axis: int, sign: int, direction: int) -> np.ndarray:
    """Permutation `perm` such that ``new_state[i] = old_state[perm[i]]`` for the
    move (axis, sign, dir). Built by labelling every sticker with its facelet
    index, geometrically rotating the affected layer, then reading where each
    label lands."""
    layer_coord = sign * (size - 1)
    # current label at each (pos, normal); rotate those in the layer.
    located: dict[tuple, int] = {}
    for f, r, c in _all_facelets(size):
        pos, normal = _facelet_to_pos_normal(size, f, r, c)
        label = f * size * size + r * size + c
        if pos[axis] == layer_coord:
            pos = _rot(pos, axis, direction)
            # rotate the outward normal vector too, then read back as (axis, sign)
            nvec = [0, 0, 0]
            nvec[normal[0]] = normal[1]
            nvec = _rot((nvec[0], nvec[1], nvec[2]), axis, direction)
            for a in (X, Y, Z):
                if nvec[a] != 0:
                    normal = (a, nvec[a])
                    break
        located[(pos, normal)] = label
    perm = np.empty(NUM_FACES * size * size, dtype=np.int64)
    for f, r, c in _all_facelets(size):
        i = f * size * size + r * size + c
        pos, normal = _facelet_to_pos_normal(size, f, r, c)
        perm[i] = located[(pos, normal)]
    return perm


# Precomputed per size: move permutations + inverse-index table.
_PERMS: dict[int, list[np.ndarray]] = {}
_INV: dict[int, list[int]] = {}


def _ensure_size(size: int) -> None:
    if size in _PERMS:
        return
    moves = moves_for(size)
    perms = [_build_perm(size, ax, sg, dr) for (_n, ax, sg, dr) in moves]
    _PERMS[size] = perms
    # inverse index: same (axis, sign), opposite dir.
    inv = []
    for (_n, ax, sg, dr) in moves:
        for j, (_n2, ax2, sg2, dr2) in enumerate(moves):
            if ax2 == ax and sg2 == sg and dr2 == -dr:
                inv.append(j)
                break
    _INV[size] = inv


# ── Public state API (operates on flat int8 facelet arrays) ─────────────────
def num_moves(size: int) -> int:
    return len(moves_for(size))


def facelet_count(size: int) -> int:
    return NUM_FACES * size * size


def state_shape(cfg: CubeConfig) -> tuple[int, ...]:
    """One-hot encoded length: every facelet (6*size^2) one-hot over 6 colors."""
    return (facelet_count(cfg.size) * NUM_COLORS,)


def solved_state(size: int) -> np.ndarray:
    s = size * size
    return np.repeat(np.arange(NUM_FACES, dtype=np.int8), s)


def apply_move(state: np.ndarray, size: int, move: int) -> np.ndarray:
    _ensure_size(size)
    return state[_PERMS[size][move]]


def inverse_move(size: int, move: int) -> int:
    _ensure_size(size)
    return _INV[size][move]


def is_solved(state: np.ndarray, size: int) -> bool:
    """Solved == every face is a single color. (For the 2x2, which has no
    centers, this accepts any whole-cube orientation, which is correct.)"""
    s = size * size
    for f in range(NUM_FACES):
        face = state[f * s:(f + 1) * s]
        if not np.all(face == face[0]):
            return False
    return True


def scramble(size: int, k: int, rng: random.Random) -> tuple[np.ndarray, list[int]]:
    """Apply `k` random moves to the solved cube, avoiding immediately undoing
    the previous move. Returns (scrambled_state, moves_applied)."""
    _ensure_size(size)
    state = solved_state(size)
    applied: list[int] = []
    m = num_moves(size)
    prev = -1
    for _ in range(k):
        choices = [mv for mv in range(m) if mv != inverse_move(size, prev)] if prev >= 0 else list(range(m))
        mv = rng.choice(choices)
        state = apply_move(state, size, mv)
        applied.append(mv)
        prev = mv
    return state, applied


def encode_batch(states: np.ndarray, size: int) -> np.ndarray:
    """One-hot encode a batch of facelet arrays -> float32 (B, 6*size^2*6)."""
    states = np.asarray(states, dtype=np.int64)
    if states.ndim == 1:
        states = states[None, :]
    onehot = np.eye(NUM_COLORS, dtype=np.float32)[states]  # (B, F, 6)
    return onehot.reshape(states.shape[0], -1)


# ── Rendering for the 3D Watch view ─────────────────────────────────────────
def render_dict(state: np.ndarray, size: int) -> dict:
    """Per-cubie sticker data for the three.js renderer. Each cubie is a small
    cube at integer coords; `stickers` lists its outward-facing colored faces."""
    cubies: dict[tuple, dict] = {}
    for f, r, c in _all_facelets(size):
        pos, normal = _facelet_to_pos_normal(size, f, r, c)
        color = int(state[f * size * size + r * size + c])
        nvec = [0, 0, 0]
        nvec[normal[0]] = normal[1]
        entry = cubies.setdefault(pos, {"pos": list(pos), "stickers": []})
        entry["stickers"].append({"normal": nvec, "color": color})
    return {
        "size": size,
        "coords": list(axis_vals(size)),
        "cubies": list(cubies.values()),
        "solved": is_solved(state, size),
    }


def move_meta(size: int) -> list[dict]:
    """Frontend-facing description of each move (for animating layer turns)."""
    out = []
    for (name, ax, sg, dr) in moves_for(size):
        out.append({"name": name, "axis": ax, "sign": sg, "dir": dr})
    return out


# ── Self-test: prove the move model is a valid cube group ───────────────────
if __name__ == "__main__":
    for size in (2, 3):
        _ensure_size(size)
        moves = moves_for(size)
        m = len(moves)
        solved = solved_state(size)
        assert is_solved(solved, size)

        # 1) each move is a bijection (valid permutation)
        for mi in range(m):
            perm = _PERMS[size][mi]
            assert sorted(perm.tolist()) == list(range(facelet_count(size))), \
                f"move {moves[mi][0]} not a permutation"

        # 2) move^4 == identity
        for mi in range(m):
            s = solved.copy()
            for _ in range(4):
                s = apply_move(s, size, mi)
            assert np.array_equal(s, solved), f"{moves[mi][0]}^4 != identity"

        # 3) move · inverse == identity
        for mi in range(m):
            s = apply_move(solved.copy(), size, mi)
            s = apply_move(s, size, inverse_move(size, mi))
            assert np.array_equal(s, solved), f"{moves[mi][0]} · inverse != identity"

        # 4) a scramble is undone by the reversed inverse sequence
        rng = random.Random(0)
        state, applied = scramble(size, 30, rng)
        assert not is_solved(state, size), "30-move scramble should not be solved"
        for mv in reversed(applied):
            state = apply_move(state, size, inverse_move(size, mv))
        assert is_solved(state, size), "reversed scramble did not solve"

        # 5) a single move actually changes the cube
        assert not is_solved(apply_move(solved.copy(), size, 0), size)

        # 6) render sanity
        rd = render_dict(solved, size)
        expected_cubies = size ** 3 - (0 if size == 2 else 1)  # 3x3 hides the core
        # (we keep all surface cubies; the 3x3 core has no stickers and is absent)
        assert len(rd["cubies"]) == (8 if size == 2 else 26), len(rd["cubies"])

        print(f"size {size}: {m} moves, {facelet_count(size)} facelets, "
              f"state dim {state_shape(CubeConfig(size))[0]} — all checks pass")
    print("OK — cube move model is a valid group for 2x2 and 3x3.")

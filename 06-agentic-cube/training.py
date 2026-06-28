"""CubeSession — the live training state both drivers (UI and RL Coach agent)
advance, plus checkpoint I/O.

Mirrors the role `training.py` plays across the suite, but the unit of work is a
**value-iteration batch** (a gradient step over freshly-scrambled states), not an
episode, and the headline metric is **solve-rate** (fraction of scrambles the
learned heuristic + beam search can solve) rather than a game score.

The reverse-scramble curriculum lives here: `train_one_iteration(k)` builds a
batch of states scrambled 1..k moves from solved, so the learnable signal grows
outward from the goal as `k` is ramped up.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import cube as C
from agents import HyperParams, build_agent, default_hyperparams
from models import CKPT_DIR

MAX_HISTORY = 5000

# Bundled, ship-with-the-app checkpoints that must not be deletable (the Watch
# tab loads these by default). Guarded in the backend so neither the UI nor the
# agent can remove them.
PROTECTED_CHECKPOINTS = {"pretrained-2x2.pt", "pretrained-3x3.pt"}


def cube_config_from_dict(d: dict | None) -> C.CubeConfig:
    d = d or {}
    return C.CubeConfig(size=int(d.get("size", 3)))


def cube_config_to_dict(cfg: C.CubeConfig) -> dict:
    return {"size": cfg.size}


class CubeSession:
    def __init__(
        self,
        algo: str,
        hyperparameters: dict,
        cube_config: dict,
        device: torch.device,
        seed: int | None = None,
    ):
        self.algo = algo
        self.hp = HyperParams(**{**default_hyperparams(algo), **(hyperparameters or {})})
        self.cube_cfg = cube_config_from_dict(cube_config)
        self.size = self.cube_cfg.size
        self.device = device
        self.in_features = C.state_shape(self.cube_cfg)[0]
        # Raises ValueError on invalid combos; callers surface it as 400 / tool error.
        self.agent = build_agent(algo, self.hp, device, self.in_features, self.size)
        self.rng = random.Random(seed)
        C._ensure_size(self.size)

        self.iteration = 0
        self.current_k = 1
        self.solve_rate_by_k: dict[int, float] = {}
        # One record per iteration: {iteration, k, loss, mean_target}.
        self.loss_history: list[dict] = []

    # ── Introspection ───────────────────────────────────────────────────────
    def param_count(self) -> int:
        return sum(p.numel() for p in self.agent.net.parameters())

    def uses_network(self) -> bool:
        return True

    # ── Curriculum scramble generation ───────────────────────────────────────
    def _scrambled_batch(self, k: int, n: int) -> np.ndarray:
        """`n` states, each scrambled a depth drawn uniformly from 1..k."""
        k = max(1, int(k))
        states = np.empty((n, C.facelet_count(self.size)), dtype=np.int8)
        for i in range(n):
            d = self.rng.randint(1, k)
            st, _ = C.scramble(self.size, d, self.rng)
            states[i] = st
        return states

    # ── Training ─────────────────────────────────────────────────────────────
    def train_one_iteration(self, k: int | None = None) -> dict:
        """One value-iteration gradient step on a fresh curriculum batch."""
        if k is None:
            k = self.current_k
        states = self._scrambled_batch(k, self.hp.batch_size)
        out = self.agent.train_batch(states)
        self.iteration += 1
        record = {
            "iteration": self.iteration,
            "k": int(k),
            "loss": round(out["loss"], 5),
            "mean_target": round(out["mean_target"], 3),
        }
        self.loss_history.append(record)
        if len(self.loss_history) > MAX_HISTORY:
            self.loss_history = self.loss_history[-MAX_HISTORY:]
        return record

    def train_iterations(
        self, n: int, k: int | None = None,
        on_iteration: Callable[[dict], None] | None = None,
    ) -> dict:
        losses: list[float] = []
        for _ in range(n):
            record = self.train_one_iteration(k)
            losses.append(record["loss"])
            if on_iteration is not None:
                on_iteration(record)
        return {
            "iterations_run": n,
            "iteration": self.iteration,
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
            "last_loss": losses[-1] if losses else 0.0,
        }

    # ── Evaluation: can it actually solve scrambles? ─────────────────────────
    def evaluate(
        self, n: int = 50, k: int | None = None,
        beam_width: int = 200, max_depth: int | None = None,
    ) -> dict:
        """Attempt `n` depth-`k` scrambles with the learned heuristic + beam
        search; report solve-rate and mean solution length."""
        if k is None:
            k = self.current_k
        if max_depth is None:
            max_depth = max(20, k * 3)
        solved = 0
        lengths: list[int] = []
        for _ in range(n):
            st, _ = C.scramble(self.size, max(1, int(k)), self.rng)
            sol = self.agent.solve(st, max_depth=max_depth, beam_width=beam_width)
            if sol is not None:
                solved += 1
                lengths.append(len(sol))
        rate = solved / n if n else 0.0
        self.solve_rate_by_k[int(k)] = round(rate, 4)
        return {
            "attempted": n,
            "k": int(k),
            "solve_rate": round(rate, 4),
            "mean_solution_len": round(sum(lengths) / len(lengths), 2) if lengths else None,
        }

    # ── Watch a single solve ─────────────────────────────────────────────────
    def solve_episode(self, k: int | None = None, beam_width: int = 400) -> dict:
        """Scramble a cube `k` moves, solve it, and record the cube state after
        every move (plus per-step move metadata + the heuristic's per-move scores)
        for the 3D Watch view. If the solver can't find a solution, replays a
        bounded greedy rollout so the user still sees the agent's behavior."""
        if k is None:
            k = self.current_k
        k = max(1, int(k))
        scrambled, scramble_moves = C.scramble(self.size, k, self.rng)
        max_depth = max(20, k * 3)
        solution = self.agent.solve(scrambled, max_depth=max_depth, beam_width=beam_width)
        solved = solution is not None
        if solution is None:
            # Bounded greedy rollout for visualization.
            solution = []
            st = scrambled.copy()
            for _ in range(max_depth):
                if C.is_solved(st, self.size):
                    break
                mv = self.agent.act(st)
                solution.append(mv)
                st = C.apply_move(st, self.size, mv)
            solved = C.is_solved(st, self.size)

        meta = C.move_meta(self.size)
        frames = [C.render_dict(scrambled, self.size)]
        steps: list[dict] = []
        st = scrambled.copy()
        for mv in solution:
            scores = self.agent.action_scores(st)
            st = C.apply_move(st, self.size, mv)
            frames.append(C.render_dict(st, self.size))
            steps.append({
                "move": meta[mv],
                "scores": scores,
            })
        return {
            "size": self.size,
            "scramble_depth": k,
            "scramble_moves": [meta[m]["name"] for m in scramble_moves],
            "frames": frames,
            "steps": steps,
            "solved": solved,
            "solution_len": len(solution),
            "move_catalog": meta,
        }

    # ── Hot-swap hyperparameters ────────────────────────────────────────────
    def update_hyperparameters(self, hp: dict) -> None:
        for k, v in (hp or {}).items():
            if hasattr(self.hp, k) and v is not None:
                setattr(self.hp, k, type(getattr(self.hp, k))(v))
        for group in self.agent.opt.param_groups:
            group["lr"] = self.hp.lr

    def move_to_device(self, device: torch.device) -> None:
        self.device = device
        self.agent.device = device
        self.agent.net.to(device)
        self.agent.target.to(device)

    # ── Checkpoint I/O ──────────────────────────────────────────────────────
    def to_checkpoint(self) -> dict:
        return {
            "algo": self.algo,
            "hyperparameters": asdict(self.hp),
            "cube_config": cube_config_to_dict(self.cube_cfg),
            "agent_state": self.agent.state_dict(),
            "iteration": self.iteration,
            "current_k": self.current_k,
            "solve_rate_by_k": self.solve_rate_by_k,
            "loss_history": self.loss_history[-MAX_HISTORY:],
        }


# ── Checkpoint helpers ──────────────────────────────────────────────────────
def _ckpt_path(filename: str) -> Path:
    if not filename.endswith(".pt"):
        filename += ".pt"
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("filename cannot contain path separators or '..'")
    return CKPT_DIR / filename


def save_checkpoint(session: CubeSession, filename: str) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(filename)
    torch.save(session.to_checkpoint(), path)
    return path.name


def load_checkpoint(filename: str, device: torch.device) -> CubeSession:
    path = _ckpt_path(filename)
    if not path.exists():
        raise FileNotFoundError(filename)
    data = torch.load(path, map_location=device, weights_only=False)
    session = CubeSession(
        data["algo"], data["hyperparameters"], data["cube_config"], device,
    )
    session.agent.load_state_dict(data["agent_state"])
    session.iteration = data.get("iteration", 0)
    session.current_k = data.get("current_k", 1)
    session.solve_rate_by_k = {int(k): v for k, v in data.get("solve_rate_by_k", {}).items()}
    session.loss_history = data.get("loss_history", [])
    return session


def list_checkpoints() -> list[dict]:
    if not CKPT_DIR.exists():
        return []
    out = []
    for p in sorted(CKPT_DIR.glob("*.pt")):
        st = p.stat()
        out.append({
            "name": p.name,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "protected": p.name in PROTECTED_CHECKPOINTS,
        })
    return out


def delete_checkpoint(filename: str) -> str:
    path = _ckpt_path(filename)
    if path.name in PROTECTED_CHECKPOINTS:
        raise ValueError(f"{path.name} is a bundled checkpoint and cannot be deleted")
    if not path.exists():
        raise FileNotFoundError(filename)
    path.unlink()
    return path.name

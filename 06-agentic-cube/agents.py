"""The cube solver agent: approximate **value iteration** with a learned
cost-to-go function (a small, local-hardware version of DeepCubeA).

The lesson of this app is *why* the model-free RL of `05` (Q-learning / DQN /
REINFORCE) can't crack the cube: reward is hopelessly sparse — one solved state
among ~4.3x10^19 — so trial-and-error exploration never finds it. The fix uses
the cube's **known, deterministic model**: from any state we can enumerate all
children, so we can bootstrap a cost-to-go target without ever needing to
stumble onto reward by luck.

Training (`train_batch`):
  * sample states from the reverse-scramble curriculum (done by the session),
  * for each state expand all `M` children and score them with the *target*
    network,
  * target cost = 0 if the state is already solved, else `1 + min_child_cost`
    (one move, plus the best child's cost-to-go),
  * regress the *online* network onto those targets (MSE), sync the target net
    every `target_update` batches.

Solving (`solve`): a beam search guided by the learned heuristic — search is what
lets an imperfect cost-to-go function still stitch together a full solution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

import cube as C
from models import CostToGoNet, pick_device


@dataclass
class HyperParams:
    """Value-iteration knobs, exposed to the UI and the RL Coach agent."""

    lr: float = 1e-3
    batch_size: int = 1000        # scrambled states per training batch
    hidden: int = 512             # width of the first hidden layer (second = hidden/2)
    target_update: int = 50       # batches between target-network syncs
    weight_decay: float = 0.0


class ValueIterationAgent:
    name = "value_iteration"

    def __init__(
        self,
        hp: HyperParams,
        device: torch.device | None,
        in_features: int,
        size: int,
    ):
        self.hp = hp
        self.device = device or pick_device()
        self.size = size
        self.in_features = in_features
        self.num_moves = C.num_moves(size)
        hidden = (hp.hidden, max(32, hp.hidden // 2))
        self.net = CostToGoNet(in_features, hidden).to(self.device)
        self.target = CostToGoNet(in_features, hidden).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
        self.updates = 0
        self._last_loss = 0.0
        self._last_mean_target = 0.0

    # ── Value helpers ────────────────────────────────────────────────────────
    def _values(self, states: np.ndarray, net: nn.Module) -> np.ndarray:
        """Cost-to-go for a batch of facelet arrays, as a numpy vector."""
        enc = C.encode_batch(states, self.size)
        t = torch.from_numpy(enc).to(self.device)
        with torch.no_grad():
            return net(t).cpu().numpy()

    def _children(self, states: np.ndarray) -> np.ndarray:
        """All children of a batch: (B, M, F) facelet arrays."""
        b = states.shape[0]
        out = np.empty((b, self.num_moves, states.shape[1]), dtype=states.dtype)
        for mi in range(self.num_moves):
            perm = C._PERMS[self.size][mi]
            out[:, mi, :] = states[:, perm]
        return out

    # ── Training ─────────────────────────────────────────────────────────────
    def train_batch(self, states: np.ndarray) -> dict:
        """One value-iteration gradient step over a batch of scrambled states."""
        C._ensure_size(self.size)
        b, f = states.shape
        children = self._children(states).reshape(b * self.num_moves, f)
        # Score children with the target net; solved children cost 0.
        child_cost = self._values(children, self.target).reshape(b, self.num_moves)
        s = self.size * self.size
        # vectorized is_solved for the children: every face uniform.
        ch = children.reshape(b, self.num_moves, C.NUM_FACES, s)
        solved_child = np.all(ch == ch[:, :, :, :1], axis=3).all(axis=2)  # (b, M)
        child_cost = np.where(solved_child, 0.0, child_cost)
        targets = 1.0 + child_cost.min(axis=1)
        # States already solved have cost 0.
        st = states.reshape(b, C.NUM_FACES, s)
        solved_state = np.all(st == st[:, :, :1], axis=2).all(axis=1)  # (b,)
        targets = np.where(solved_state, 0.0, targets).astype(np.float32)

        enc = torch.from_numpy(C.encode_batch(states, self.size)).to(self.device)
        tgt = torch.from_numpy(targets).to(self.device)
        pred = self.net(enc)
        loss = nn.functional.mse_loss(pred, tgt)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.updates += 1
        if self.updates % self.hp.target_update == 0:
            self.sync_target()
        self._last_loss = float(loss.item())
        self._last_mean_target = float(targets.mean())
        return {"loss": self._last_loss, "mean_target": self._last_mean_target}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    # ── Acting / inspection ──────────────────────────────────────────────────
    def cost_to_go(self, state: np.ndarray) -> float:
        return float(self._values(state[None, :], self.net)[0])

    def act(self, state: np.ndarray) -> int:
        """Greedy: the move whose child has the lowest predicted cost-to-go."""
        children = self._children(state[None, :])[0]  # (M, F)
        costs = self._values(children, self.net)
        for mi in range(self.num_moves):
            if C.is_solved(children[mi], self.size):
                return mi
        return int(np.argmin(costs))

    def action_scores(self, state: np.ndarray) -> dict | None:
        """Per-move resulting cost-to-go for the Watch overlay (lower is better)."""
        children = self._children(state[None, :])[0]
        costs = self._values(children, self.net)
        return {"kind": "cost", "values": [round(float(x), 3) for x in costs]}

    def solve(self, state: np.ndarray, max_depth: int = 40, beam_width: int = 200) -> list[int] | None:
        """Beam search guided by the learned cost-to-go. Returns a move list that
        reaches a solved state, or None if not found within the budget."""
        if C.is_solved(state, self.size):
            return []
        seen: set[bytes] = {state.tobytes()}
        # beam entries: (state_array, moves_so_far)
        beam: list[tuple[np.ndarray, list[int]]] = [(state, [])]
        for _ in range(max_depth):
            cand_states: list[np.ndarray] = []
            cand_moves: list[list[int]] = []
            for st, moves in beam:
                for mi in range(self.num_moves):
                    child = C.apply_move(st, self.size, mi)
                    if C.is_solved(child, self.size):
                        return moves + [mi]
                    key = child.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    cand_states.append(child)
                    cand_moves.append(moves + [mi])
            if not cand_states:
                break
            costs = self._values(np.array(cand_states), self.net)
            order = np.argsort(costs)[:beam_width]
            beam = [(cand_states[i], cand_moves[i]) for i in order]
        return None

    # ── Persistence ──────────────────────────────────────────────────────────
    @property
    def metrics(self) -> dict:
        return {
            "loss": round(self._last_loss, 5),
            "mean_target": round(self._last_mean_target, 3),
            "updates": self.updates,
        }

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, data: dict) -> None:
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(self.net.state_dict())
        if "opt" in data:
            self.opt.load_state_dict(data["opt"])
        self.updates = data.get("updates", 0)


# ── Registry ────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "value_iteration": {
        "label": "Value Iteration (cost-to-go)",
        "uses_network": True,
        "description": (
            "Learns a cost-to-go heuristic (moves-to-solve) using the cube's "
            "known model: expand every child, bootstrap targets from a target "
            "network, regress. Trained on a reverse-scramble curriculum and "
            "paired with beam search at solve time — the DeepCubeA idea, scaled "
            "to local hardware."
        ),
    },
}


def build_agent(
    algo: str,
    hp: HyperParams | None,
    device: torch.device | None,
    in_features: int,
    size: int,
) -> ValueIterationAgent:
    hp = hp or HyperParams()
    if algo == "value_iteration":
        return ValueIterationAgent(hp, device, in_features, size)
    raise ValueError(f"unknown algorithm: {algo}")


def default_hyperparams(algo: str) -> dict:
    base = HyperParams()
    return asdict(base)

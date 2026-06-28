"""The three RL agents, behind one shared interface.

All three see the same 11-feature state and choose among the same 3 relative
actions, so they are directly comparable. They differ in *how* they learn:

* **QLearningAgent** — tabular. A lookup table of Q-values, updated by the
  Bellman rule. No neural network: the value of every (state, action) is a
  number you can read directly. The "what is a Q-value" lesson.
* **DQNAgent** — deep value-based. A neural net approximates Q, trained off a
  replay buffer with a slowly-updated target network. The canonical deep-RL
  algorithm.
* **ReinforceAgent** — policy gradient. A neural net outputs the policy
  directly and is nudged, at the end of each episode, toward actions that led
  to higher return. Value-based vs. policy-based, side by side.

The training loop (in `training_worker.py` / `train.py`) only ever touches the
shared interface:

    a = agent.act(state)                       # choose
    next_state, reward, done, _ = env.step(a)
    agent.observe(state, a, reward, next_state, done)   # learn (per-step agents)
    ...
    agent.end_episode()                        # learn (episodic agents)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import NUM_ACTIONS, STATE_SIZE, SnakeEnv
from models import PolicyNetwork, QNetwork, pick_device


@dataclass
class HyperParams:
    """Tunable knobs shared (where meaningful) across the agents.

    Not every agent uses every field — REINFORCE ignores epsilon (its policy
    is stochastic by construction), tabular Q ignores the network width — but
    keeping one struct means the UI and the RL Coach agent have a single
    surface to read and write.
    """

    lr: float = 1e-3
    gamma: float = 0.9               # discount on future reward
    epsilon_start: float = 1.0       # exploration rate at episode 0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99      # per-episode multiplicative decay
    hidden: int = 128                # MLP width (deep agents)
    batch_size: int = 64             # DQN replay minibatch
    buffer_size: int = 50_000        # DQN replay capacity
    target_update: int = 500         # DQN steps between target-net syncs
    warmup: int = 1_000              # DQN steps of pure exploration before learning


class BaseAgent:
    """The interface the training loop relies on. Subclasses fill in the learning."""

    name: str = "base"

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        raise NotImplementedError

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> None:
        """Per-step hook. Value-based agents learn here; REINFORCE just records."""

    def end_episode(self) -> None:
        """End-of-episode hook. Episodic agents (REINFORCE) learn here."""

    def action_scores(self, state: np.ndarray) -> dict | None:
        """Per-action scores for the Watch-tab overlay, so the user can see
        what the agent 'thinks'. `{"kind": "q"|"prob", "values": [...]}` —
        Q-values for the value-based agents, probabilities for REINFORCE.
        None when the agent has nothing to show (e.g. an unseen tabular state)."""
        return None

    @property
    def metrics(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, data: dict) -> None:
        pass


# ── Tabular Q-learning ──────────────────────────────────────────────────────
class QLearningAgent(BaseAgent):
    name = "qlearning"

    def __init__(self, hp: HyperParams, obs_shape: tuple[int, ...] = (STATE_SIZE,)):
        if tuple(obs_shape) != (STATE_SIZE,):
            raise ValueError(
                "Tabular Q-learning needs the engineered 'features' observation — "
                "the full grid has astronomically many states for a lookup table "
                "(that's exactly why we need function approximation). Use DQN or "
                "REINFORCE with the grid observation instead."
            )
        self.hp = hp
        self.q: dict[tuple, np.ndarray] = {}
        self.epsilon = hp.epsilon_start
        self._last_td: float = 0.0

    def _key(self, state: np.ndarray) -> tuple:
        # The 11 features are 0/1, so the state is naturally a discrete key.
        return tuple(int(v) for v in state)

    def _row(self, key: tuple) -> np.ndarray:
        row = self.q.get(key)
        if row is None:
            row = np.zeros(NUM_ACTIONS, dtype=np.float32)
            self.q[key] = row
        return row

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        return int(np.argmax(self._row(self._key(state))))

    def observe(self, state, action, reward, next_state, done) -> None:
        row = self._row(self._key(state))
        target = reward
        if not done:
            target += self.hp.gamma * float(np.max(self._row(self._key(next_state))))
        td = target - row[action]
        row[action] += self.hp.lr * td
        self._last_td = abs(float(td))

    def end_episode(self) -> None:
        self.epsilon = max(self.hp.epsilon_min, self.epsilon * self.hp.epsilon_decay)

    def action_scores(self, state: np.ndarray) -> dict | None:
        key = self._key(state)
        if key not in self.q:
            return None  # never-seen state — no learned values yet
        return {"kind": "q", "values": self.q[key].tolist()}

    @property
    def metrics(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "td_error": self._last_td,
            "q_states": len(self.q),
        }

    def state_dict(self) -> dict:
        # Tuple keys aren't JSON-friendly but pickle (torch.save) handles them.
        return {"q": {k: v.tolist() for k, v in self.q.items()}, "epsilon": self.epsilon}

    def load_state_dict(self, data: dict) -> None:
        self.q = {k: np.asarray(v, dtype=np.float32) for k, v in data["q"].items()}
        self.epsilon = data.get("epsilon", self.hp.epsilon_min)


# ── Deep Q-Network ──────────────────────────────────────────────────────────
class DQNAgent(BaseAgent):
    name = "dqn"

    def __init__(
        self, hp: HyperParams, device: torch.device | None = None,
        obs_shape: tuple[int, ...] = (STATE_SIZE,),
    ):
        self.hp = hp
        self.device = device or pick_device()
        self.q = QNetwork(obs_shape, hp.hidden).to(self.device)
        self.target = QNetwork(obs_shape, hp.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=hp.lr)
        self.buffer: deque[tuple] = deque(maxlen=hp.buffer_size)
        self.epsilon = hp.epsilon_start
        self.steps = 0
        self._last_loss: float = 0.0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        with torch.no_grad():
            t = torch.from_numpy(state).to(self.device).unsqueeze(0)
            return int(self.q(t).argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        self.steps += 1
        if len(self.buffer) >= max(self.hp.batch_size, self.hp.warmup):
            self._learn()
        if self.steps % self.hp.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

    def _learn(self) -> None:
        batch = random.sample(self.buffer, self.hp.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).to(self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_sa = self.q(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            best_next = self.target(next_states).max(dim=1).values
            target = rewards + self.hp.gamma * best_next * (1.0 - dones)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._last_loss = float(loss.item())

    def end_episode(self) -> None:
        self.epsilon = max(self.hp.epsilon_min, self.epsilon * self.hp.epsilon_decay)

    def action_scores(self, state: np.ndarray) -> dict | None:
        with torch.no_grad():
            t = torch.from_numpy(state).to(self.device).unsqueeze(0)
            return {"kind": "q", "values": self.q(t).squeeze(0).cpu().tolist()}

    @property
    def metrics(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "loss": self._last_loss,
            "buffer": len(self.buffer),
        }

    def state_dict(self) -> dict:
        return {
            "q": self.q.state_dict(),
            "opt": self.opt.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }

    def load_state_dict(self, data: dict) -> None:
        self.q.load_state_dict(data["q"])
        self.target.load_state_dict(self.q.state_dict())
        if "opt" in data:
            self.opt.load_state_dict(data["opt"])
        self.epsilon = data.get("epsilon", self.hp.epsilon_min)
        self.steps = data.get("steps", 0)


# ── REINFORCE (Monte-Carlo policy gradient) ─────────────────────────────────
class ReinforceAgent(BaseAgent):
    name = "reinforce"

    def __init__(
        self, hp: HyperParams, device: torch.device | None = None,
        obs_shape: tuple[int, ...] = (STATE_SIZE,),
    ):
        self.hp = hp
        self.device = device or pick_device()
        self.policy = PolicyNetwork(obs_shape, hp.hidden).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=hp.lr)
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._last_loss: float = 0.0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        t = torch.from_numpy(state).to(self.device).unsqueeze(0)
        if greedy:
            with torch.no_grad():
                return int(self.policy(t).argmax(dim=1).item())
        logits = self.policy(t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

    def observe(self, state, action, reward, next_state, done) -> None:
        # REINFORCE is Monte-Carlo: just bank the reward; learning waits for
        # the full episode return.
        self._rewards.append(reward)

    def end_episode(self) -> None:
        if not self._log_probs:
            return
        # Discounted return-to-go at each step.
        returns: list[float] = []
        g = 0.0
        for r in reversed(self._rewards):
            g = r + self.hp.gamma * g
            returns.insert(0, g)
        ret = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Baseline via standardization keeps gradients well-scaled.
        if len(ret) > 1:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        log_probs = torch.cat(self._log_probs)
        loss = -(log_probs * ret).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._last_loss = float(loss.item())
        self._log_probs.clear()
        self._rewards.clear()

    def action_scores(self, state: np.ndarray) -> dict | None:
        with torch.no_grad():
            t = torch.from_numpy(state).to(self.device).unsqueeze(0)
            probs = F.softmax(self.policy(t), dim=1).squeeze(0)
            return {"kind": "prob", "values": probs.cpu().tolist()}

    @property
    def metrics(self) -> dict:
        return {"loss": self._last_loss}

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, data: dict) -> None:
        self.policy.load_state_dict(data["policy"])
        if "opt" in data:
            self.opt.load_state_dict(data["opt"])


# ── Registry ────────────────────────────────────────────────────────────────
# Metadata the CLI, the server, and the RL Coach agent all read from.
ALGORITHMS = {
    "qlearning": {
        "label": "Tabular Q-learning",
        "uses_network": False,
        "description": (
            "A lookup table of Q-values updated by the Bellman rule. No neural "
            "network — every (state, action) value is a number you can inspect."
        ),
    },
    "dqn": {
        "label": "Deep Q-Network (DQN)",
        "uses_network": True,
        "description": (
            "A neural net approximates Q-values, trained off a replay buffer "
            "with a slowly-updated target network. The canonical deep-RL method."
        ),
    },
    "reinforce": {
        "label": "REINFORCE (policy gradient)",
        "uses_network": True,
        "description": (
            "A policy network is nudged, after each episode, toward actions that "
            "led to higher return. Learns the policy directly, not values."
        ),
    },
}


def build_agent(
    algo: str, hp: HyperParams | None = None, device: torch.device | None = None,
    obs_shape: tuple[int, ...] = (STATE_SIZE,),
) -> BaseAgent:
    hp = hp or HyperParams()
    if algo == "qlearning":
        return QLearningAgent(hp, obs_shape)
    if algo == "dqn":
        return DQNAgent(hp, device, obs_shape)
    if algo == "reinforce":
        return ReinforceAgent(hp, device, obs_shape)
    raise ValueError(f"unknown algorithm: {algo}")


def default_hyperparams(algo: str) -> dict:
    """Per-algorithm sensible starting hyperparameters (as a plain dict)."""
    base = HyperParams()
    if algo == "qlearning":
        base.lr = 0.1
        base.gamma = 0.9
    elif algo == "dqn":
        base.lr = 1e-3
        base.gamma = 0.9
    elif algo == "reinforce":
        base.lr = 1e-3
        base.gamma = 0.95
    return asdict(base)

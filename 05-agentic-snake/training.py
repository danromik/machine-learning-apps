"""SnakeSession — the live training state both drivers (UI and RL Coach
agent) advance, plus checkpoint I/O.

Mirrors the role `training.py` plays in `04-agentic-symbols`, but for RL: the
unit of work is an *episode* (a full game) rather than a gradient batch, and
the headline metric is *score* (apples eaten) rather than loss/accuracy.

One session owns an environment + an agent + the rolling history of episode
outcomes. `train_episodes(n, on_episode=...)` runs episodes synchronously and
calls back per episode so the caller can stream progress to the UI;
`play_episode()` runs a single greedy game and records every frame (plus the
agent's per-action scores) for the Watch tab.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch

from agents import HyperParams, build_agent, default_hyperparams
from game import OBSERVATIONS, EnvConfig, RewardConfig, SnakeEnv, state_shape
from models import CKPT_DIR

# Keep history bounded so long runs don't grow without limit (mirrors the
# frontend chart cap).
MAX_HISTORY = 5000


def env_config_from_dict(d: dict | None) -> EnvConfig:
    d = d or {}
    r = d.get("reward", {}) or {}
    observation = str(d.get("observation", "features"))
    if observation not in OBSERVATIONS:
        observation = "features"
    return EnvConfig(
        width=int(d.get("width", 10)),
        height=int(d.get("height", 10)),
        observation=observation,
        reward=RewardConfig(
            food=float(r.get("food", 1.0)),
            death=float(r.get("death", -1.0)),
            step=float(r.get("step", 0.0)),
            toward_food=float(r.get("toward_food", 0.0)),
            away_from_food=float(r.get("away_from_food", 0.0)),
        ),
    )


def env_config_to_dict(cfg: EnvConfig) -> dict:
    return {
        "width": cfg.width,
        "height": cfg.height,
        "observation": cfg.observation,
        "reward": {
            "food": cfg.reward.food,
            "death": cfg.reward.death,
            "step": cfg.reward.step,
            "toward_food": cfg.reward.toward_food,
            "away_from_food": cfg.reward.away_from_food,
        },
    }


class SnakeSession:
    def __init__(
        self,
        algo: str,
        hyperparameters: dict,
        env_config: dict,
        device: torch.device,
    ):
        self.algo = algo
        self.hp = HyperParams(**{**default_hyperparams(algo), **(hyperparameters or {})})
        self.env_cfg = env_config_from_dict(env_config)
        self.device = device
        self.obs_shape = state_shape(self.env_cfg)
        # Raises ValueError for invalid combos (e.g. tabular Q-learning on the
        # grid); callers surface that as a 400 / tool error.
        self.agent = build_agent(algo, self.hp, device, self.obs_shape)
        self.env = SnakeEnv(self.env_cfg)
        self.episode = 0
        self.best_score = 0
        # One record per episode: {episode, score, reward, length, ...metrics}.
        self.score_history: list[dict] = []

    # ── Introspection ───────────────────────────────────────────────────────
    def param_count(self) -> int:
        net = getattr(self.agent, "q", None) or getattr(self.agent, "policy", None)
        if net is None:  # tabular
            return 0
        return sum(p.numel() for p in net.parameters())

    def uses_network(self) -> bool:
        return self.param_count() > 0

    # ── Training ────────────────────────────────────────────────────────────
    def train_one_episode(self) -> dict:
        """Play and learn from a single episode; return its record. The
        per-episode unit both drivers (UI chunk-loop, agent tool) build on."""
        state = self.env.reset()
        total_reward = 0.0
        while not self.env.done:
            action = self.agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.observe(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        self.agent.end_episode()
        self.episode += 1
        self.best_score = max(self.best_score, self.env.score)
        record = {
            "episode": self.episode,
            "score": self.env.score,
            "reward": round(total_reward, 3),
            "length": self.env.steps,
            **{k: (round(v, 5) if isinstance(v, float) else v)
               for k, v in self.agent.metrics.items()},
        }
        self.score_history.append(record)
        if len(self.score_history) > MAX_HISTORY:
            self.score_history = self.score_history[-MAX_HISTORY:]
        return record

    def train_episodes(
        self, n: int, on_episode: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run `n` episodes, learning each one. Calls `on_episode(record)`
        after each. Returns a summary across the run."""
        scores: list[int] = []
        for _ in range(n):
            record = self.train_one_episode()
            scores.append(record["score"])
            if on_episode is not None:
                on_episode(record)
        return {
            "episodes_run": n,
            "episode": self.episode,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "best_score": self.best_score,
            "last_score": scores[-1] if scores else 0,
        }

    def evaluate(self, n: int = 20) -> dict:
        """Greedy evaluation — exploration off. Doesn't learn."""
        scores: list[int] = []
        lengths: list[int] = []
        for _ in range(n):
            state = self.env.reset()
            while not self.env.done:
                action = self.agent.act(state, greedy=True)
                state, _, _, _ = self.env.step(action)
            scores.append(self.env.score)
            lengths.append(self.env.steps)
        return {
            "episodes": n,
            "mean_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "mean_length": sum(lengths) / len(lengths),
        }

    # ── Watch a single game ─────────────────────────────────────────────────
    def play_episode(self, greedy: bool = True, max_frames: int = 2000) -> dict:
        """Run one episode and record every frame + the agent's per-action
        scores, for the Watch tab to animate."""
        state = self.env.reset()
        frames: list[dict] = [self.env.render_dict()]
        steps: list[dict] = []
        while not self.env.done and len(frames) < max_frames:
            scores = self.agent.action_scores(state)
            action = self.agent.act(state, greedy=greedy)
            state, reward, done, info = self.env.step(action)
            frames.append(self.env.render_dict())
            steps.append({
                "action": action,
                "reward": round(reward, 3),
                "scores": scores,
                "event": info.get("event"),
            })
        return {
            "frames": frames,
            "steps": steps,
            "score": self.env.score,
            "length": self.env.steps,
        }

    # ── Hot-swap hyperparameters ────────────────────────────────────────────
    def update_hyperparameters(self, hp: dict) -> None:
        """Apply live hyperparameter changes. lr updates the optimizer in
        place for the deep agents; the rest take effect on the next step."""
        for k, v in (hp or {}).items():
            if hasattr(self.hp, k) and v is not None:
                setattr(self.hp, k, type(getattr(self.hp, k))(v))
        opt = getattr(self.agent, "opt", None)
        if opt is not None:
            for group in opt.param_groups:
                group["lr"] = self.hp.lr

    def move_to_device(self, device: torch.device) -> None:
        self.device = device
        self.agent.device = device  # type: ignore[attr-defined]
        for attr in ("q", "target", "policy"):
            net = getattr(self.agent, attr, None)
            if net is not None:
                net.to(device)

    # ── Checkpoint I/O ──────────────────────────────────────────────────────
    def to_checkpoint(self) -> dict:
        from dataclasses import asdict
        return {
            "algo": self.algo,
            "hyperparameters": asdict(self.hp),
            "env_config": env_config_to_dict(self.env_cfg),
            "agent_state": self.agent.state_dict(),
            "episode": self.episode,
            "best_score": self.best_score,
            "score_history": self.score_history[-MAX_HISTORY:],
        }


# ── Checkpoint helpers ──────────────────────────────────────────────────────
def _ckpt_path(filename: str) -> Path:
    if not filename.endswith(".pt"):
        filename += ".pt"
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("filename cannot contain path separators or '..'")
    return CKPT_DIR / filename


def save_checkpoint(session: SnakeSession, filename: str) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(filename)
    torch.save(session.to_checkpoint(), path)
    return path.name


def load_checkpoint(filename: str, device: torch.device) -> SnakeSession:
    path = _ckpt_path(filename)
    if not path.exists():
        raise FileNotFoundError(filename)
    data = torch.load(path, map_location=device, weights_only=False)
    session = SnakeSession(
        data["algo"], data["hyperparameters"], data["env_config"], device,
    )
    session.agent.load_state_dict(data["agent_state"])
    session.episode = data.get("episode", 0)
    session.best_score = data.get("best_score", 0)
    session.score_history = data.get("score_history", [])
    return session


def list_checkpoints() -> list[dict]:
    if not CKPT_DIR.exists():
        return []
    out = []
    for p in sorted(CKPT_DIR.glob("*.pt")):
        st = p.stat()
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


def delete_checkpoint(filename: str) -> str:
    path = _ckpt_path(filename)
    if not path.exists():
        raise FileNotFoundError(filename)
    path.unlink()
    return path.name

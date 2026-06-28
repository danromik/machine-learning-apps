"""The Snake environment and its 11-feature state encoder.

This is the *environment* half of the RL loop: it owns the grid, the snake,
the food, and the rules. An agent picks one of three relative actions each
step (go straight / turn right / turn left) and the env returns the next
state, a reward, and whether the episode ended.

Two design choices worth knowing:

* **Relative actions.** The action space is {straight, right, left} relative
  to the snake's current heading — not the four absolute directions. This
  makes a 180-degree suicide impossible and makes the learned policy
  orientation-invariant, which is much easier (and faster) to learn.

* **Two observation models.** `EnvConfig.observation` selects what the agent
  sees:

  - `"features"` (default) — a small hand-engineered vector (danger x3,
    heading x4, food-direction x4). It is discrete and tiny, so tabular
    Q-learning is feasible and the deep agents stay small and fast. Its
    weakness: it only probes the three cells next to the head, so the agent
    is blind to the shape of its own body and eventually traps itself.
  - `"grid"` — the full board as a (channels, height, width) tensor (snake
    body, snake head, food). The agent can finally *see its whole body*, at
    the cost of a much larger state: tabular Q-learning is no longer feasible
    (that's the canonical motivation for function approximation), so this
    mode is for the deep agents (DQN, REINFORCE) with a small CNN.

  All agents still choose among the same 3 relative actions, so the two deep
  algorithms remain directly comparable across observation models.

`models.py`, `agents.py`, `training.py`, and `train.py` import the env and
`STATE_SIZE` / `NUM_ACTIONS` / `state_shape` from here.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ── Geometry ────────────────────────────────────────────────────────────────
# y increases downward (row index), x increases rightward (column index).
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Relative actions, in a fixed order the agents index into.
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
ACTION_NAMES = ["straight", "right", "left"]

NUM_ACTIONS = 3
STATE_SIZE = 11        # length of the engineered feature vector

# Grid observation: one plane each for snake-body, snake-head, and food.
GRID_CHANNELS = 3
OBSERVATIONS = ("features", "grid")


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    """Rotate a heading 90 degrees clockwise (screen coords, y down)."""
    dx, dy = d
    return (-dy, dx)


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    """Rotate a heading 90 degrees counter-clockwise (screen coords, y down)."""
    dx, dy = d
    return (dy, -dx)


def _apply_action(d: tuple[int, int], action: int) -> tuple[int, int]:
    if action == TURN_RIGHT:
        return _turn_right(d)
    if action == TURN_LEFT:
        return _turn_left(d)
    return d  # STRAIGHT


@dataclass
class RewardConfig:
    """The reward components, all live-tunable from the UI / agent.

    Reward shaping is the heart of RL pedagogy, so every term is exposed:
    a big signal for eating and dying, plus optional shaping that nudges the
    snake toward food and discourages aimless wandering.
    """

    food: float = 1.0           # eating an apple
    death: float = -1.0         # hitting a wall or itself
    step: float = 0.0           # per-step living cost (negative discourages stalling)
    toward_food: float = 0.0    # bonus for moving closer to food (Manhattan)
    away_from_food: float = 0.0  # penalty for moving farther (use a negative value)


@dataclass
class EnvConfig:
    width: int = 10
    height: int = 10
    start_length: int = 3
    reward: RewardConfig = field(default_factory=RewardConfig)
    # What the agent observes each step: "features" (11-vector) or "grid"
    # (full board tensor). See the module docstring for the trade-off.
    observation: str = "features"
    # Hard cap on episode length so a perfect survivor can't run forever.
    max_steps: int = 0            # 0 -> derived from grid area below
    # Truncate (no death penalty) if the snake goes this long without eating;
    # kills the "circle forever" degenerate policy. 0 -> derived from area.
    max_steps_without_food: int = 0


def state_shape(cfg: EnvConfig) -> tuple[int, ...]:
    """The shape of the observation `cfg` produces — `(STATE_SIZE,)` for the
    engineered features, `(GRID_CHANNELS, height, width)` for the grid. The
    model builder uses this to size the network (MLP vs. CNN)."""
    if cfg.observation == "grid":
        return (GRID_CHANNELS, cfg.height, cfg.width)
    return (STATE_SIZE,)


class SnakeEnv:
    """A single Snake game. `reset()` starts an episode, `step(action)` advances it."""

    def __init__(self, config: EnvConfig | None = None, seed: int | None = None):
        self.cfg = config or EnvConfig()
        self.rng = random.Random(seed)
        self._area = self.cfg.width * self.cfg.height
        self.reset()

    # ── Derived caps ────────────────────────────────────────────────────────
    @property
    def _max_steps(self) -> int:
        return self.cfg.max_steps or (self._area * 4)

    @property
    def _max_steps_without_food(self) -> int:
        return self.cfg.max_steps_without_food or (self._area * 2)

    # ── Episode lifecycle ───────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        cx, cy = self.cfg.width // 2, self.cfg.height // 2
        self.direction = RIGHT
        # Body laid out head-first; tail extends left of the head.
        length = max(1, min(self.cfg.start_length, cx + 1))
        self.body: deque[tuple[int, int]] = deque(
            (cx - i, cy) for i in range(length)
        )
        self.score = 0           # apples eaten == snake length grown
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        occupied = set(self.body)
        free = [
            (x, y)
            for x in range(self.cfg.width)
            for y in range(self.cfg.height)
            if (x, y) not in occupied
        ]
        # No free cell -> the snake fills the board (a perfect game).
        self.food = self.rng.choice(free) if free else None

    # ── Collision helpers ───────────────────────────────────────────────────
    def _hits_wall(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return not (0 <= x < self.cfg.width and 0 <= y < self.cfg.height)

    def _would_die(self, cell: tuple[int, int]) -> bool:
        """True if moving the head onto `cell` next step is fatal.

        The tail vacates as the snake moves, so colliding with the current
        tail tip is allowed (unless the snake is about to grow, which only
        happens when it eats — and food cells are never fatal anyway).
        """
        if self._hits_wall(cell):
            return True
        occupied = set(self.body)
        occupied.discard(self.body[-1])  # tail will move out of the way
        return cell in occupied

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        self.direction = _apply_action(self.direction, action)
        hx, hy = self.body[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        self.steps += 1
        r = self.cfg.reward

        # Death by wall or self.
        if self._would_die(new_head):
            self.done = True
            return self.get_state(), r.death, True, {"event": "death"}

        eating = new_head == self.food
        # Distance shaping is computed before we move the head.
        prev_dist = self._food_distance()

        self.body.appendleft(new_head)
        if eating:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
        else:
            self.body.pop()
            self.steps_since_food += 1

        reward = r.step
        if eating:
            reward += r.food
        else:
            new_dist = self._food_distance()
            if new_dist < prev_dist:
                reward += r.toward_food
            elif new_dist > prev_dist:
                reward += r.away_from_food

        # Truncation: filled the board (win) or wandered too long, or hit cap.
        info: dict = {"event": "eat" if eating else "move"}
        if self.food is None:
            self.done = True
            info["event"] = "win"
        elif self.steps_since_food >= self._max_steps_without_food:
            self.done = True
            info["event"] = "timeout"
            info["truncated"] = True
        elif self.steps >= self._max_steps:
            self.done = True
            info["truncated"] = True

        return self.get_state(), reward, self.done, info

    # ── State encoding ──────────────────────────────────────────────────────
    def _food_distance(self) -> int:
        if self.food is None:
            return 0
        hx, hy = self.body[0]
        fx, fy = self.food
        return abs(hx - fx) + abs(hy - fy)

    def get_state(self) -> np.ndarray:
        """Encode the current state per `cfg.observation`."""
        if self.cfg.observation == "grid":
            return self._grid_state()
        return self._feature_state()

    def _grid_state(self) -> np.ndarray:
        """The full board as a (GRID_CHANNELS, height, width) float32 tensor.

        Channel 0 = snake body (excluding the head), 1 = head, 2 = food. The
        agent sees every cell, so it can perceive the whole shape of its body
        — unlike the engineered features. Coordinates are (x, y) with y the
        row, so we index ``grid[channel, y, x]``.
        """
        w, h = self.cfg.width, self.cfg.height
        grid = np.zeros((GRID_CHANNELS, h, w), dtype=np.float32)
        for (x, y) in self.body:
            grid[0, y, x] = 1.0
        hx, hy = self.body[0]
        grid[0, hy, hx] = 0.0   # head lives in its own channel, not the body one
        grid[1, hy, hx] = 1.0
        if self.food is not None:
            fx, fy = self.food
            grid[2, fy, fx] = 1.0
        return grid

    def _feature_state(self) -> np.ndarray:
        """The 11-feature vector: danger x3, heading x4, food-direction x4."""
        hx, hy = self.body[0]
        d = self.direction
        dir_r = _turn_right(d)
        dir_l = _turn_left(d)

        def ahead(delta: tuple[int, int]) -> tuple[int, int]:
            return (hx + delta[0], hy + delta[1])

        danger_straight = self._would_die(ahead(d))
        danger_right = self._would_die(ahead(dir_r))
        danger_left = self._would_die(ahead(dir_l))

        moving_up = d == UP
        moving_down = d == DOWN
        moving_left = d == LEFT
        moving_right = d == RIGHT

        if self.food is not None:
            fx, fy = self.food
            food_left = fx < hx
            food_right = fx > hx
            food_up = fy < hy
            food_down = fy > hy
        else:
            food_left = food_right = food_up = food_down = False

        return np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                moving_up,
                moving_down,
                moving_left,
                moving_right,
                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )

    # ── Rendering ───────────────────────────────────────────────────────────
    def render_dict(self) -> dict:
        """A JSON-serializable snapshot for the frontend game board."""
        return {
            "width": self.cfg.width,
            "height": self.cfg.height,
            "snake": [list(c) for c in self.body],
            "food": list(self.food) if self.food else None,
            "score": self.score,
            "steps": self.steps,
            "done": self.done,
        }


# ── Sanity check: a random agent runs episodes end-to-end ───────────────────
if __name__ == "__main__":
    env = SnakeEnv(EnvConfig(width=10, height=10), seed=0)
    n_episodes = 5
    rng = random.Random(0)
    for ep in range(n_episodes):
        state = env.reset()
        assert state.shape == (STATE_SIZE,), state.shape
        total_reward = 0.0
        while not env.done:
            action = rng.randrange(NUM_ACTIONS)
            state, reward, done, info = env.step(action)
            total_reward += reward
        print(
            f"episode {ep}: score={env.score} steps={env.steps} "
            f"reward={total_reward:+.2f} last={info['event']}"
        )
    print("OK — SnakeEnv runs episodes and encodes an 11-feature state.")

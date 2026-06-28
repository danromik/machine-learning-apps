"""Headless CLI trainer — the ground truth for the Snake RL pipeline.

Fastest way to confirm an agent actually learns before any server/UI exists:

    uv run python 05-agentic-snake/train.py --algo qlearning --episodes 400
    uv run python 05-agentic-snake/train.py --algo dqn --episodes 400 --device cpu
    uv run python 05-agentic-snake/train.py --algo reinforce --episodes 600

Reports a rolling-mean score (apples eaten) so you can watch learning happen,
then runs a greedy evaluation with exploration turned off.
"""

from __future__ import annotations

import argparse
from collections import deque

import torch

from agents import ALGORITHMS, HyperParams, build_agent, default_hyperparams
from game import OBSERVATIONS, EnvConfig, RewardConfig, SnakeEnv, state_shape


def run_episode(env: SnakeEnv, agent, greedy: bool = False) -> tuple[int, float, int]:
    state = env.reset()
    total_reward = 0.0
    while not env.done:
        action = agent.act(state, greedy=greedy)
        next_state, reward, done, _ = env.step(action)
        if not greedy:
            agent.observe(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    if not greedy:
        agent.end_episode()
    return env.score, total_reward, env.steps


def main() -> None:
    p = argparse.ArgumentParser(description="Train a Snake RL agent (headless).")
    p.add_argument("--algo", choices=list(ALGORITHMS), default="qlearning")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--grid", type=int, default=10, help="square grid side length")
    p.add_argument("--observation", choices=list(OBSERVATIONS), default="features",
                   help="what the agent sees: 'features' (11-vector) or 'grid' (full board)")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toward-food", type=float, default=0.0,
                   help="reward shaping: bonus for moving toward food")
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = None if args.device == "auto" else torch.device(args.device)

    reward = RewardConfig(toward_food=args.toward_food,
                          away_from_food=-args.toward_food)
    env_cfg = EnvConfig(width=args.grid, height=args.grid, reward=reward,
                        observation=args.observation)
    env = SnakeEnv(env_cfg, seed=args.seed)

    hp = HyperParams(**default_hyperparams(args.algo))
    agent = build_agent(args.algo, hp, device, state_shape(env_cfg))

    print(f"Training {ALGORITHMS[args.algo]['label']} for {args.episodes} episodes "
          f"on a {args.grid}x{args.grid} grid (obs={args.observation})"
          + (f" (device={agent.device})" if hasattr(agent, "device") else "")
          + "\n")

    window: deque[int] = deque(maxlen=50)
    best = 0
    report_every = max(1, args.episodes // 20)
    for ep in range(1, args.episodes + 1):
        score, _, _ = run_episode(env, agent)
        window.append(score)
        best = max(best, score)
        if ep % report_every == 0 or ep == 1:
            avg = sum(window) / len(window)
            m = agent.metrics
            extra = ""
            if "epsilon" in m:
                extra += f" eps={m['epsilon']:.3f}"
            if "loss" in m:
                extra += f" loss={m['loss']:.4f}"
            if "q_states" in m:
                extra += f" states={m['q_states']}"
            print(f"ep {ep:4d}  avg_score(50)={avg:5.2f}  best={best}{extra}")

    # Greedy evaluation — exploration off, this is the policy you'd deploy.
    eval_scores = [run_episode(env, agent, greedy=True)[0]
                   for _ in range(args.eval_episodes)]
    avg_eval = sum(eval_scores) / len(eval_scores)
    print(f"\nGreedy eval over {args.eval_episodes} episodes: "
          f"avg_score={avg_eval:.2f}  best={max(eval_scores)}")


if __name__ == "__main__":
    main()

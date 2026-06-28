# 3×3 Cube — Final Debrief

**Algorithm:** value iteration over a learned cost-to-go function (DeepCubeA-lite), ~298K params, MPS
**Method:** reverse-scramble curriculum (k=1 → k=14) + beam search at solve time
**Final session:** ~4,400+ value-iteration iterations

## Solve-rate by scramble depth (fresh final eval, 80 cubes/depth)
| k (moves from solved) | 7 | 10 | 13 | 14 |
|---|---|---|---|---|
| **solve-rate** | 100% | 100% | 83.75% | 72.5% |
| mean solution length | 6.9 | 9.5 | 15.1 | 20.7 |

Shallow-to-mid depths (≤10) are **fully mastered**, and the earlier curriculum
checkpoints recorded ≥92% all the way through k=12. The frontier is k=13–14, where
the solver is good but no longer near-perfect.

## Is the cube genuinely solvable now?
**Yes — for real, within the depths it was trained on.** This is not a heuristic
that "kind of points the right way": at k≤10 it solves *every* scrambled cube, and
the solutions are near-optimal (k=10 scrambles solved in ~9.5 moves on average —
shorter than the scramble, because beam search cancels redundant moves). At k=13 it
still solves ~5 of every 6 cubes. The deepest depth attempted (k=14) drops to ~73%
and the solutions get noticeably longer (~21 moves) — search is taking detours
where the heuristic is weaker.

## Key decisions and what they taught
- **Why not plain model-free RL?** A scrambled 3×3 has one goal among ~4.3×10¹⁹
  states. Random exploration essentially *never* hits it, so Q-learning/DQN would
  flail. The motivating failure of the whole project.
- **Exploit the model instead.** The cube is deterministic and reversible, so from
  any state we can enumerate all children and learn a **cost-to-go** (moves-to-solve)
  by bootstrapping each state's value from its children. No reward-stumbling required.
- **Reverse-scramble curriculum.** We can't *explore* to states near the goal, but
  we can *generate* them by scrambling the solved cube k moves. Train shallow, ramp k
  as each depth clears 90%. We watched it auto-advance k=1→13 — the central lesson of
  the run.
- **Heuristic + search, not heuristic alone.** The learned cost-to-go is imperfect,
  so beam search stitches the full solution. Learned heuristic + search is the trick.

## Limits hit
- **Diminishing returns with depth.** Each extra scramble move multiplies the
  reachable state space, so a fixed batch of iterations buys 100% at k=7 and almost
  nothing at k=13. Late training showed **flat loss (~0.15) with solve-rate barely
  moving** — the sparse-reward difficulty made concrete, not a bug.
- **Didn't fully clear k=13–14.** k=13 never crossed the 90% promotion bar, so the
  curriculum correctly held there; k=14 is best-effort at ~73%.
- **Capacity/time.** ~298K params and a few thousand iterations is enough for
  shallow-to-mid mastery, not for God's-number-deep (20-move) scrambles.

## Honest next steps
- **Overnight `start_training_run`** (resumes from checkpoint): thousands more
  iterations would push k=13 past 90% and grind k=14 upward — with diminishing
  speed. The 3×3 is best-effort by design.
- **Want a guaranteed 100%-at-every-depth solver?** The **2×2** (~3.6M states)
  fully converges in minutes — a complete end-to-end solve.
- **Try this model now** on the **Watch** tab: scrambles ≤10 solve cleanly and
  near-optimally; push to 13–14 to see search occasionally fail or take the long way.
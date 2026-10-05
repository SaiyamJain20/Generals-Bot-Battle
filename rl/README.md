# RL track

Overnight RL experiments (2–3 Oct 2026). They were developed in a separate git worktree (branch `rl-track`) by a
second Claude Code session working in parallel with the heuristic work, and later merged here under `rl/`.

- `rl/hier/`, `rl/final/`: Track A, RO-PPO (residual option policy over the heuristic's candidate options).
- `rl/students/`, `rl/py*`: Track B, tiny behaviour-cloned conv policies compiled to pure Python (too weak).
- `rl/c/`: Track C, GRPO/PPO and ES over the heuristic's context modulator, plus the official-evaluator parity check.
- `rl/DESIGN.md`: design and evaluation protocol. `rl/board.md`: timestamped log. `rl/agents/`: research reports.

Result: no RL candidate beat the heuristic submission (F2) under a fair test. See the top-level README, section 7.

## Submission constraints (hard)
- One UTF-8 Python 3.12 file, standard library only (no numpy/torch), <= 1 MiB, exposing
  act(observation) -> [kind, row, col, dir, split].
- 150 ms per move (10 s for the first, incl. import) on ONE CPU shared by both bots, 2 GiB RAM.
- Timeout / exception / malformed return = forfeit.
- Boards 18-21 x 18-21, fog of war, castles, deathtouch from turn 800, draw at 1200.
- Engine pinned: vendor/generals-bots @13db8f69; exact stdlib replica in sim/engine.py (tests/test_parity.py).

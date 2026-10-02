# RL track (worktree `rl-track`)

Coordinator: the main Claude session working in this worktree. A separate session keeps improving the
heuristic bot in the main checkout; the heuristic stays as the fallback until an RL bot beats it head-to-head.

## Rules for agents
- Work ONLY inside this worktree: /home/saiyamjain/Desktop/Bot-Battle/.claude/worktrees/rl (write under rl/).
  Never edit anything in the main checkout /home/saiyamjain/Desktop/Bot-Battle (outside .claude/worktrees).
- Report file: rl/agents/<agent-name>.md. One-line progress: append to rl/board.md (`HH:MM | name | note`).
- Do not run version-control commands (the coordinator commits). No GPU on this laptop. Never touch the Ada cluster.
- The laptop is shared with another session's game evaluations: use at most 2 CPU cores, keep RSS < 2 GB,
  check `free -h` before heavy work. Kill only processes you started, by PID.
- Python: .venv312/bin/python (symlink; Python 3.12 with numpy, torch CPU, jax CPU), PYTHONPATH=vendor/generals-bots:.

## Submission constraints (hard)
- One UTF-8 Python 3.12 file, standard library only (no numpy/torch), <= 1 MiB, exposing
  act(observation) -> [kind, row, col, dir, split].
- 150 ms per move (10 s for the first, incl. import) on ONE CPU shared by both bots, 2 GiB RAM.
- Timeout / exception / malformed return = forfeit.
- Boards 18-21 x 18-21, fog of war, castles, deathtouch from turn 800, draw at 1200.
- Engine pinned: vendor/generals-bots @13db8f69; exact stdlib replica in sim/engine.py (tests/test_parity.py).

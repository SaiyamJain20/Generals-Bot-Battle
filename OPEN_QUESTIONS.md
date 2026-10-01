# Open questions (still unanswered)

Update this file as answers arrive. Resolved items move to the bottom.

| # | Question | Who / how | Status |
|---|---|---|---|
| 1 | **Starter kit**: where are the event evaluator, `starter.py`, `pass_bot.py` and the dict adapter that calls `act(observation)`? Exact grid format (nested lists `grid[r][c]` or flat)? Exact keys? Are values plain `int`? | Ask organizer | OPEN |
| 2 | **Ada access**: key-based SSH (`ssh ada` alias) or run jobs manually? Account/QoS limits: run `sacctmgr show assoc user=$USER format=account,qos%30` and `sinfo -o "%P %l %D %c %m %G"`. How to move code (scp/rsync vs private git remote)? | You | OPEN |
| 3 | **Event image**: exact Python 3.12.x patch, how the 150 ms is measured (includes observation delivery), is stderr allowed/visible, rough CPU speed? | Ask organizer | OPEN |
| 4 | **Timing**: exact H0 and deadline (with timezone), submission link | Ask organizer | OPEN |
| 5 | **Identity**: participant ID (file must be named `participant_id.py`) and bot name for the header | You | OPEN |
| 6 | Any strong/RL bot we can play against before the deadline? (Assumed no; public generals.bot replays stand in) | Ask organizer | OPEN |

## Resolved
- Training on public generals.bot replays: **OK** (user confirmed, 2026-10-02). Attribute in file header.
- Build window: 30 h for this user (PDF says 20 h).
- Core approach: heuristic + exact sim tactics + CMA-ES tuning + small learned eval (user choice).

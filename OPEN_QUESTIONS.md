# Open questions (still unanswered)

Update this file as answers arrive. Resolved items move to the bottom.

| # | Question | Who / how | Status |
|---|---|---|---|
| 1 | **Starter kit**: where are the event evaluator, `starter.py`, `pass_bot.py` and the dict adapter that calls `act(observation)`? Exact grid format (nested lists `grid[r][c]` or flat)? Exact keys? Are values plain `int`? | Ask organizer | OPEN |
| 3 | **Event image**: exact Python 3.12.x patch, how the 150 ms is measured (includes observation delivery), is stderr allowed/visible, rough CPU speed? | Ask organizer | OPEN |
| 4 | **Timing**: exact H0 and deadline (with timezone), submission link | Ask organizer | OPEN |
| 5 | **Identity**: participant ID (file must be named `participant_id.py`) and bot name for the header | You | OPEN |
| 6 | Any strong/RL bot we can play against before the deadline? (Assumed no; public generals.bot replays stand in) | Ask organizer | OPEN |

## Resolved
- Ada access: key SSH as guest `<cluster_user>@<login_node>` works; limits research/low = 10 CPU, 1 GPU, 32 GB, 5 jobs, 4 days. Code moved by rsync to `~/botbattle-saiyam` (rules in CLAUDE.md, every created path/job in `ada_manifest.txt`).
- Local GPU: **not to be used** (user instruction, 2026-10-02); all GPU training runs on Ada.
- Training on public generals.bot replays: **OK** (user confirmed, 2026-10-02). Attribute in file header.
- Build window: 30 h for this user (PDF says 20 h).
- Core approach: heuristic + exact sim tactics + CMA-ES tuning + small learned eval (user choice).

## New (2026-10-02 midday)
| # | Question | Who | Status |
|---|---|---|---|
| 7 | May we build and run the third-party C++ bot "A9" (mortid0/generals-bots, Marathon branch, `atlas-negative-candidate.zip`, already downloaded to vendor/ext/mortid0_a9) as a LOCAL sparring partner? The permission classifier blocked building it. | You | RESOLVED: user approved building A9 (2026-10-02); built as bots/opp/ext_a9.py (weak, local only) |
| 8 | Participant ID and bot name for the submission file name and header (`participant_id.py`) | You | OPEN |

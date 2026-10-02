# Shared workspace for parallel agents (Bot-Battle)

Coordinator: the main Claude session. Read `CLAUDE.md` (Ada rules) and the plan
(`~/.claude/plans/so-i-have-given-luminous-adleman.md`) before starting.

## Protocol
- Each agent owns ONE file: `agents_shared/<agent-name>.md` (status, findings, how-to-run).
  Write findings there as you go (append, newest at the bottom), so others can read them.
- One-line progress updates go to `agents_shared/board.md` (append only):
  `HH:MM | <agent-name> | <what changed / where>`.
- Shared questions/ideas list: `agents_shared/questions.md` (append; mark answered ones with ✅ and a pointer).

## Hard rules for all agents
- Do NOT edit `bots/participant.py`, `bots/versions/*`, `sim/engine.py`, `arena/run.py`, `tune/*` (coordinator owns them).
- New opponent bots go in `bots/opp/ext_<name>.py` (single file exposing `act(observation)`; may import from
  `vendor/ext/<repo>/` and may use numpy/torch). Observation = dict with keys turn, height, width, player_id,
  my_land, my_army, opp_land, opp_army, type, owner, army (grids are nested lists [r][c]; owner 1 = me, 2 = enemy;
  type 0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure-in-fog). Return [kind,row,col,dir,split].
- External repos are cloned into `vendor/ext/<repo>/` (gitignored). Record URL + licence in your agent file.
  External code is for LOCAL TESTING ONLY; it must never go into the submission.
- CPU is shared with long tuning jobs: use at most 3 worker processes for any batch of games.
  Use `.venv312/bin/python` (Python 3.12). GPU: prefix commands with `CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=`.
- Never touch the Ada cluster.
- Test a new opponent with: `.venv312/bin/python tools/ab4.py bots/opp/ext_<name>.py bots/opp/hunter.py 20 3`
  and against our bot: `.venv312/bin/python tools/ab4.py bots/versions/t1c.py bots/opp/ext_<name>.py 20 3`.

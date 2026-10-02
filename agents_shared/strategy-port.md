# strategy-port — porting the best readable bots' strategies into our bot

Owner: strategy-port agent. Status: IN PROGRESS.
Base: `bots/versions/sp_base.py` = `bots/participant.py` + `runs/ada2/best.json` (made with tools/make_variant.py).
Variants: `bots/versions/sp_<idea>.py` (sp_base + one change behind a PARAMS flag, flag ON).
A/B: `tools/abmulti.py 40 3 --bots ... --opps bots/opp/ext_sentinel10.py bots/versions/t1c.py bots/opp/zoo_sniper.py --offset 3000`
Raw outputs: `runs/sp/*.txt`.

## Log (newest at bottom)

### Study summary (what the readable top bots do that we don't)
- **Sentinel v10** (vendor/ext/relh_generals-bots, chain V10>V9>V8>V6>V5>v2 scoring in sentinel_v3_agent.py):
  - castles: no turn window/cadence; builds on ANY non-structure stack with a >= price+5+max adjacent enemy,
    home path dist >= 2, home not threatened, t+2*price+100 < 1200 (v3.py:130-145). No dedicated gathering.
  - offence: V8 "collection": rally 2-3 steps from target, owned route <= 6 steps, delivered = sum(a-1) along it
    >= target + counter (max adjacent enemy) + eta + 1; V9 remembers the general and keeps pushing toward it.
  - defence: V5 corridor interception (attacker's cheapest path home, defenders that reach an own corridor cell
    before it with enough army; only off-corridor donors) + V6 commitment for 10 turns. Author's lessons:
    concentrated force > land; gathering against cheap targets wastes moves; dropping a defence when the
    threat goes into fog loses games.
- **bca heuristics** (boss/human_exe): boss builds only with no credible threat, general NOT visible, cell
  army >= cost+12, <= 4 castles, t < 680; defence = garrison max(4+fog, threat+3), intercept with a stack >=
  threat+2 else screen. RL notes: builds were good mainly when the builder was BEHIND; filter "t<=700, behind,
  post-build garrison >= 10, enemy > 3 away".
- **Our games** (sp_base vs Sentinel/t1c, instrumented with a scratch smoke script):
  - vs Sentinel we win (5/5 smoke) although Sentinel owns 90-125 land vs our ~40-60 by t300-500; we lose
    20-50 tiles per 100 turns to its roaming stacks.
  - kill attempts (try_kill) fire 14-67 moves per game but fail: e.g. map 3005 vs t1c, estimate 23 (seen 2
    turns earlier) yet the true general jumps 30 -> 69 when our ~65 stack is 13 steps into a 20-step visible
    run; t333 estimate 27+growth vs true 81. Enemy reinforces the general while it watches our long approach
    (replay rule: top bots strike from <= 8 cells, a ~6-turn run).
  - idle army sits on the general (t200: gen 60 vs need 8) and in stale strike remainders near the front
    (44-59 army at enemy-dist 1-3) which our build rule never uses (window 244-434, de >= 5).

### Experiments (40 games/opp, offset 3000, 3 workers; pooled +-95% CI ~0.075)
| variant | change | sentinel10 | t1c | zoo_sniper | pooled |
|---|---|---|---|---|---|
| sp_base | participant.py + ada2 best | 0.75 | 0.80 | 0.725 | 0.758 |
| sp_sweep | `x_sweep`: kill check walks the path, own cells join the stack (Sentinel V8 collection), 10 candidates | 0.725 | 0.775 | 0.85 | 0.783 |
| sp_stage | sweep + `x_stage`: launch within 8 of known general that cannot kill re-gathers in place (<=3x10 moves) + Sentinel counter term | 0.725 | 0.725 | 0.825 | 0.758 |

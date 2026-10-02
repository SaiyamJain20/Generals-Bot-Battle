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
| sp_corridor | `x_corridor`: Sentinel V5/V6 corridor interception (see below) | 0.725 | 0.825 | 0.80 | 0.783 |
| sp_opbuild | `x_opbuild`: Sentinel opportunistic castle rule outside our window / closer than castle_safe_dist | 0.675 | 0.675 | 0.725 | 0.692 |

Confirmation at offset 3100 (40 games/opp):
| variant | sentinel10 | t1c | zoo_sniper | pooled |
|---|---|---|---|---|
| sp_base | 0.75 | 0.825 | 0.775 | 0.783 |
| sp_sweep | 0.75 | 0.80 | 0.75 | 0.767 |
| sp_corridor | 0.75 | 0.80 | 0.775 | 0.775 |
| sp_opbuild | 0.75 | 0.80 | 0.775 | 0.775 |

**Combined over both offsets (80 games/opp, 240 games, CI about +-0.053):**
base 0.771 | sweep 0.775 | corridor 0.779 | opbuild 0.733 | stage 0.758 (only the 120 games at offset 3000, where base was also 0.758).

### Idea details and verdicts
1. **Kill conversion: `x_sweep` (+ optional `x_stage`).**
   - `try_kill` walks the Dijkstra path exactly: own cells join the stack, one army is left per step, and every
     non-own cell must be beaten. Kill if arrival >= gen_est + len/2 + kill_margin + 1. It considers 10
     candidate stacks instead of 4.
   - Result: neutral locally, 0.775 vs 0.771. It fires more kill runs. Keep the patch: it is cheap and correct.
   - `x_stage` adds two things on top. A launch that is within 8 of the known general but cannot kill yet
     re-gathers where it stands (up to 3 times, 10 moves each). The kill check also adds Sentinel's counter
     term (the strongest enemy cell next to the general).
   - Result: neutral, 0.758 = base on 120 games. Optional patch.
   - The real conversion failure we found, the enemy reinforcing while watching our long visible run, is not
     fully fixed by either. A follow-up worth trying is a stealth path for try_kill runs longer than about 8.
2. **Defence vs a big stack: `x_corridor`.**
   - Threat test: a visible enemy cell with A >= 10 within 10 steps, where A - D[e] - steps/2 > 0.5*A[g]. D
     is a home-cost Dijkstra from the general: 1 + our army per path cell.
   - Corridor = the cheapest path home. Donors are off-corridor own stacks that reach an own corridor cell c_m
     in at most m-1 steps (own-cell BFS) with A[s]-1+A[c] >= attacker army on arrival + 2. Sentinel score:
     100*ds + 10*m.
   - The donor's route is committed for 10 turns, even into fog. The option score is
     max(w_build, w_garrison_urgent) + 0.5.
   - It fires rarely against the local pool (0-10 moves a game) because our garrison already keeps A[g] high.
     The clones' winning mechanism, one unsplit 60-100 stack, is exactly what it targets.
   - Result: neutral-positive locally, 0.779 vs 0.771. **Recommend cluster test vs the clones.**
3. **Economy: `x_opbuild` (Sentinel castle rule).**
   - Builds on any idle, non-reserved own plain stack with A >= price + 5 + the strongest adjacent enemy,
     dist_g >= 2, enemy distance >= 2, t >= 100, t + 2*price + 100 < 1200, home not threatened. It only applies
     where our own rule does not fire (outside 244-434, or enemy distance < 5).
   - Result: 0.733 vs 0.771, negative on both offsets' sum (t1c 0.74 vs 0.81). **Rejected.**
   - This matches the earlier finding that castles cost us, and the bca RL note (builds pay mainly when
     behind). The stale stacks near the front are better used as the next strike's core than as forward
     castles.

### Patches (unified diffs against bots/participant.py md5 c98fff2e, flags ON; set the flag to 0 to disable)
- `agents_shared/patches/sp_1_sweep.diff`: x_sweep (keeps the coordinator's `self.mod["kill"]` multiplier).
- `agents_shared/patches/sp_2_corridor.diff`: x_corridor (+ corr_* params, `corridor_defense()`).
- `agents_shared/patches/sp_3_stage.diff`: x_sweep + x_stage.
- `agents_shared/patches/sp_4_combo_sweep_corridor.diff`: sweep + corridor together.
  - Bot file: `bots/versions/sp_combo.py`.
  - Smoke test: 3/3 wins, err 0. Two games vs Sentinel as player 1; one with current participant + diff vs t1c,
    in which corridor fired 10 times.
- All variants were smoke-tested with DEBUG=True: `_BOT.err == 0`, max 12 ms/move.

Status: DONE.

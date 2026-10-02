# castle-study: castle keeping and the land war in top-bot replays

Status: DONE (2026-10-02, 17:50-18:10).

**Scripts** (in `agents_shared/castle_study/`):
- `study.py`: castle, drain and land-war records.
- `study2.py`: opening moves and army-distribution snapshots.
- `analyze.py`, `analyze2.py`, `analyze3.py`, `cycle.py`: the tables.

**How to run**
```
.venv312/bin/python agents_shared/castle_study/study.py 4
.venv312/bin/python agents_shared/castle_study/analyze.py [all|top|weak]
```
- Each extraction takes 12-45 s on 4 workers.
- Outputs are `data/castle_study_recs.jsonl` (65 MB) and `data/castle_study_recs2.jsonl` (10 MB).
- They are **not gitignored**: do not commit them, and delete them when done.

**Data**
- Actions come from `data/acts.jsonl` (exact parse, games with fails==0); grids come from the replays.
- Sample sizes (player-games): ResBot, nanomena and Kubic 500 each; FreeLunch 250; bca and Chig 400 each. study2 uses 300 per bot.
- **Visible** means 3x3 vision around own cells (engine rule).
- φ = turn % 50. The land bonus (+1 on every owned cell) lands right after the move made at φ=49.

## Q1 Castle keeping: top bots do NOT garrison castles; they harvest them
- **Army after build.** They build on a cell holding exactly ~35-37, so the castle starts with 1-3 army.
- **Castle army at build +5/+10/+20/+40/+80** (median, still-owned castles): 3 / 6 / 10 / 12-13 / 12.
- **Median castle army over its whole life:** 11 (IQR 6-19). This is the same for top and weak bots.
- **Move-outs.** 2.1-2.3 move-outs per castle per 100 ticks, i.e. one every ~45 ticks, carrying a median of 27 army (IQR 17-50).
  - 98% are full drains (leave 1). Only 2% are splits.
  - Destination: 92-94% go to own land, 4% go into enemy land.
  - Drains are spread over the whole bonus cycle, slightly more at φ 20-40.
- **Threat at the moment of a full drain** (max visible enemy cell army, median): within 3 cells 0, within 5 cells 0, within 8 cells 3.
  - Visible enemy within 5 is ≥10 in only 8-9% of drains, and ≥20 in 6%.
- **P(castle lost within 10 / 20 ticks after a full drain), by visible enemy max within 3 (th3), ResBot:**

| th3 | P(lost ≤10 t) | P(lost ≤20 t) |
|---|---|---|
| 0-2 | 0.01 | 0.02 |
| 3-9 | 0.03 | 0.07 |
| 10-19 | 0.07 | 0.11 |
| 20-39 | 0.13 | 0.15 |
| ≥40 | 0.22 | 0.28 |

  nanomena and Kubic look the same; bca is 0.06/0.13 already at th3 3-9.
- **When an enemy stack ≥10 is within 3 and bigger than the castle,** they almost never reinforce:
  - median castle army change over 5 ticks is +2 (natural growth);
  - reinforced (≥+5): 3-5%;
  - drained: 10-12%;
  - P(lost within 20) is only 8-9%. Most enemy stacks next to a castle do not attack it.
  - When they do drain under threat, 42% of those drains go INTO the enemy (counter-attack) and 57% go to own land.
- **Coverage vs a visible enemy within 5 (th5),** per castle-tick:
  - th5 ≥ 10: castle/threat ratio median 0.33, and the castle covers the threat only 16-23% of the time.
  - Being covered still helps:
    - th5 20-39: P(lost ≤20) is 0.07-0.18 if covered vs 0.23-0.25 if not;
    - th5 10-19: 0.05-0.07 vs 0.06-0.12.

## Q2 Castle loss: big stacks take them, and top bots take them back fast
- **Loss rate** (excluding the final flip): top 35-38%, FreeLunch 48%, bca/Chig 51-52%.
  - Against non-top opponents, the top bots lose 27-32%.
  - Median age at loss is 80-90 ticks (IQR 30-190).
- **The attacking stack is big:** moved amount median 40 (IQR 23-70) against a castle army of 11 (IQR 7-18).
  - Attacker/castle ratio median 3.4. In 78% of losses the attacker is ≥2x the castle, and in 40% it is ≥4x.
  - Only 13-18% of losses happen with castle army ≤5, so a garrison would not have saved most castles.
- **Warning.** The attacker was visible within 5 for a median of 5 ticks: a straight walk-in.
- **Recent drains.** 37-47% of lost castles had been drained (to 1) in the 20 ticks before.
- **Owner's stacks at the moment of loss.**
  - Biggest own stack: median 52 at distance 6.
  - Nearest own stack ≥ the attacker: distance 8-13 (none exists in 37-46% of losses).
  - Army ratio at loss: 0.93-0.97.
- **Aftermath: the decisive difference.**
  - The enemy usually walks on through the castle: enemy army on it is 24-28 at capture and 4-9 a few ticks later.
  - Top bots retake the castle in a median of 5 ticks (IQR 3-10).
  - The retake uses a stack of ~25 (IQR 11-50) against a castle army of ~6, coming from cells ~7 from their own general.

| | recaptured ≤10 t | ≤30 t | ever |
|---|---|---|---|
| ResBot / nanomena / Kubic | 76-79% | 95-97% | 97-99% |
| bca / Chig | 68% | 86-90% | 89-92% |
| top vs non-top opponents | 85-86% | ~100% | — |

- **Build site matters far more than garrison.**
  - P(lost) by distance from the castle to the nearest enemy cell at build (de), top bots:

| de | 0-3 | 4-6 | 7-9 | ≥10 |
|---|---|---|---|---|
| top bots | 0.57 | 0.42-0.45 | 0.21-0.28 | 0.13 |
| weak bots | 0.74-0.78 | 0.60-0.63 | — | — |

  - Castles built ≥21 from the enemy general: 0.18-0.25 vs 0.47-0.52 at ≤16.
  - Early builds (before T120) are rare: 57-152 castles per 500 games, lost 25-57% of the time.

## Q3 Land war (median per game; mean in parentheses)
- **Enemy tiles captured per 50 ticks** (ResBot; nanomena and Kubic are within ±0.5):

| | T50-100 | T100-150 | T150-200 | T200-300 | T300-400 |
|---|---|---|---|---|---|
| ResBot captures | 6 (6.2) | 5 (5.7) | 7 (7.1) | 8 (8.9) | 10 (10.5) |
| bca captures | 4.9 | 4.8 | 6.4 | 8.4 | 8.3 |

  - Losses mirror the captures (peer games): about 5.5 / 5.5 / 7 / 8.5 / 9.5. Failed attacks are ≤1 per 50 ticks.
  - Total for T50-200: ~19 captured, ~18 lost.
  - Versus us, the clone makes ~22 moves into our land (about its normal capture rate) but we make only ~13 into its land. **The gap is mostly OUR offence (13 vs ~19), not our defence.**
- **Timing: the clearest pattern in the data.** Enemy captures are back-loaded in each 50-tick bonus cycle.
  - Mean captures per 5-tick φ bin, T150-200 (ResBot): 0, 0, .1, .2, .3, .5, .5, 1.0, 1.9, 2.7.
  - φ < 15: ~0. φ 35-49: 74-78% of all enemy captures. Every bot does this, weak ones included.
  - Gathering (own-target moves with amount ≥2) peaks at φ 0-25.
  - Neutral 1-army spreading peaks at φ 0-5.
- **Stack sizes** (moved amount) for enemy captures:

| | 2-3 | 4-6 | 7-15 | 16-40 | 41+ |
|---|---|---|---|---|---|
| T50-150 | 20% | 21% | 36% | 21% | 1% |
| T150-300 | 17% | 11% | 19% | 29% | 21% |

  - Target army is 1 in 80% of captures at T50-150 and 50% at T150-300.
  - Captured tile position: median 11 from own general, 9-10 from the enemy general (just past the midline).
- **Chains** (the same stack moving on consecutive ticks, with at least one enemy capture), ResBot, per 50 ticks:

| first-move amount m0 | T50-100 | T150-200 | T200-300 | typical chain |
|---|---|---|---|---|
| 2-5 | 0.5 | 1.3 | 1.9 | 1 capture, length 1: a single nibble from a frontier cell, not a walking raid |
| 6-15 (the main raid) | 1.2-1.3 | 1.1 | 1.0 | 10 moves (IQR 4-14) with 3 enemy captures (IQR 2-4) at T50-100; later 5 moves and 2 captures |
| 16-40 | 0.2 | 0.6 | 1.1 | 2-3 captures |
| ≥41 | ~0 | 0.1 | 0.4 | — |

  - Share of T50-200 enemy captures: m0 2-5 = 23-28%, m0 6-15 = 47-51%, m0 ≥16 = 22-29%.
  - Weak bots make fewer m0 6-15 raids (0.6-0.9 per 50 ticks).
- **Recapture of own tiles the enemy took within distance 5 of the general:** NOT a priority.
  - T0-200: about 1.8 events per game. Time to recapture: median 30-37 ticks (IQR 11-50). P(≤5 t) 8-11%, P(≤10 t) 19%, never 14-19%.
  - T200-400: median 14-17 ticks.
  - These tiles are simply retaken by the next cycle's spread or gather.

## Q4 How they reach ~50 land at T100 and ~70-76 at T300
- **Bonus spend-down.** At t=50 there are 22 frontier cells holding 2 army. By t=75 only 1-3 remain.
  - All bonus army is spread into neutral within ~6-10 ticks: T50-56 is almost entirely 1-army moves into neutral (5.3 per 6 ticks for ResBot).
  - Opening totals: T50-100 has 24 neutral captures (41% with 1-army moves, 47% with stacks ≥4), plus 6 enemy captures, from 50 moves and 0 passes.
- **Army is spent to near zero before each bonus.** Army vs land right before the bonus (top bots, median):

| tick | 99 | 149 | 199 | 249 | 299 |
|---|---|---|---|---|---|
| army / land | 55 / 49 | 87 / 61 | 113-118 / 64-67 | 151 / 72 | 194 / 76 |

  - At t99, 95% of land holds 1 army.
  - Max stack: 5 (t99), 10-14 (t149), 14-18 (t199), 20 (t249), 25 (t299).
  - The biggest own stack is <20 for the whole of T75-250, except when gathering 35 for a castle.
  - Mid-cycle (t125, t175), 43-51% of land holds 2 (the unharvested bonus).
  - The general holds 4-14 army and is emptied about once every 25 ticks (move of 10-20).
- **The army sits at the front.**
  - 38-46% of all army is within 3 steps of enemy land from T150 on.
  - Front cells: 9-15 own cells touching enemy land; 55-90% of them hold 1 army just before the bonus, on both sides.
  - The border is thin on both sides, so whoever brings more 6-15 raids wins tiles.
- **Castle cadence.** Castles at ~T123 / 167 / 212 (replay-analyst) add 3 × 0.5 army/tick, i.e. +75 army per 50 ticks from T220. That matches the whole land bonus of ~70 land.
- **"Expand with small stacks, then gather" is not the difference.** The difference is:
  1. phase-locking the attacks to φ 35-49;
  2. ~1.2 walking raids of 6-15 per 50 ticks from T50 on;
  3. never holding unspent army before T250;
  4. castles from T120 on, retaken within ~5 ticks when lost.

## Implementable rules (concrete parameters)
1. **Attack phase-lock** (biggest, cheap). Let φ = turn % 50.
   - φ 0-24: gather. Harvest the +1 on each 2-army interior cell toward 1-3 raid stacks near the front. Spread 1-army moves into neutral at φ 0-8.
   - φ 25-49: launch enemy-tile captures, ramping up so ~75% happen at φ ≥ 35. Multiply enemy-capture values by ~2 when φ ≥ 35 and by ~0.3 when φ < 15.
   - Today `bonus_mult` / `bonus_window=12` boosts only neutral captures at φ<12, and enemy captures get no timing at all.
2. **Raids.** Form stacks of 6-15 at frontier cells and walk them through enemy 1-army tiles (each costs 2 army).
   - Target ~1.2 raids per 50 ticks in T50-200, each 2-4 captures. Add ~1-2 single nibbles per 50 ticks from 2-5 frontier stacks (rising to 2 at T200+).
   - Overall target: ≥6 enemy-tile captures per 50 ticks in T50-150 and ≥7-9 in T150-300.
3. **Spend-down.** Before each bonus, unspent army (total army minus land) should be ≤ ~10 at t99, ~25 at t149, ~45 at t199.
   - Max stack < 20 through T250, except while building a castle.
   - Measure our bot with: `army - land` at t=99/149/199, and max stack at t=149/199.
4. **Castles: harvest, do not garrison.**
   - Drain fully (leave 1) about every 45 ticks when visible enemy max within 3 is <10.
   - With th3 10-19, a half drain is fine.
   - With th3 ≥20, do not drain into own land. Either keep castle army ≥ th3+1 when that costs ≤ ~10 extra, or drain INTO the threatening stack / enemy tiles (top bots: 42% of threatened drains).
   - Expected effect: P(lost ≤20) drops from ~0.25 to ~0.1 in the th5 20-39 band. The current `castle_keep=1` rule (always half) gives away harvest for little gain: 72% of castle-ticks have th5 ≤2, where the loss rate is 1-2%.
5. **Castle site.** Prefer cells whose nearest enemy-owned cell is ≥7 away (loss 0.14-0.22 vs 0.50-0.56 at ≤4) and that are ≥17-21 from the enemy general, at price 35 (distance 7 from own general). Avoid builds before ~T120.
6. **Castle recapture.** When an own castle flips, make retaking it the top non-defence goal within 5 ticks.
   - Send the nearest own stack ≥ castle army + 1. The castle is typically at 4-9 army a few ticks after capture, and the retaking stack is ~25.
   - Targets: retake ≤10 ticks ≥75% of the time, and ≥95% eventually.
7. **Near-general tile recapture is low priority** (top median 30 ticks). Do not divert raid stacks for it unless a stack ≥30 is involved (see replay-analyst §4).

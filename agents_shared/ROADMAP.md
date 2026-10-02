# Improvement roadmap (coordinator-maintained)

Updated 2026-10-02 06:20. Sources: training-research.md, bot-scout.md, zoo-builder.md, porter.md,
algo-study.md, replay-analyst.md, strategy-research.md, vast-research.md.

## Where we stand
- The tuned bot `t1c` beats every public/external bot we could wrap: superbot / juraj34 (C++), Sentinel,
  Amin PPO, hv-nguyeen, mybot9, doomstack, the bca heuristics. Scores run 0.80–1.00.
- It also beats the 7-bot heuristic zoo, scoring 0.63–0.88.
- The hardest opponent is the behaviour-cloned ResBot clone, at about 0.53–0.70. Bigger clones of
  ResBot, nanomena and Kubic are training on Ada (job bb-bc).
- Our accounting (exact enemy castle builds and price rings) was validated 28/28 on real games.
- The general belief never prunes the true general.

## Decisions taken from the research
| Idea | Status |
|---|---|
| Self-play + league snapshots + PFSP(1.0) in CMA-ES | **running** on Ada (ada3: league-every 8, max 6, self-play; zoo + BC + old versions as anchors) |
| Generic libraries (OpenSpiel/PettingZoo) | not useful beyond our exact engine (training-research) |
| External bots as sparring partners | done (porter): ~15 `ext_*` bots. All weaker than t1c, so used for catastrophe-gate testing |
| Heuristic zoo of distinct styles | done (zoo-builder): 7 `zoo_*` bots in the tuning pool |
| Early land expansion to ~50 land at T100 (replay benchmark) | implemented (`early_expand_until`); A/B on Ada (eval struct1) |
| Castle #1 at dist 7, enemy-facing | implemented (`castle_front_w`); A/B on Ada (eval struct1) |
| Radius-r ring counted as defence | implemented (`ring_w`, `ring_r`); A/B on Ada (eval struct1) |
| Stealth routing for strikes | implemented (`stealth_w`); A/B on Ada (eval stealth) |
| Lead-conditioned aggression | implemented (`lead_w`); A/B on Ada (eval stealth) |
| Learned threat GBM (garrison) | implemented (`learned_threat_w`); local A/B showed no gain, so it stays off; left in the CMA space |
| EklipZ fog gather queue | implemented (`fog_model`, `fog_tracks`); offline check worse than 0.3×hidden bound, so it stays off |
| Chase-kill interceptor, drain veto, capture leaks, castle gating, belief bound, sweep-DP kill | **experimenter** A/B-ing on copies → patches in `agents_shared/patches/` |

## Next candidates (ranked)
1. Ship the **averaged CMA mean** (last 5–10 gens), not the best-of-generation sample. Confirm on fresh maps plus holdouts (vast-research #2, training-research #3).
2. **Catastrophe gate** before any submission: ≥ 0.95 vs every simple bot (pass/random/expander/hunter/zoo/ext), 0 forfeits, 2–3× slow-CPU stress in `arena/subproc.py`.
3. **Contact-triggered 2-ply search**, using the exact sim against a small enemy reply set, near stacks and the general (strategy #3, vast #8).
4. **Exploiter loop**: CMA-tune a copy against the frozen current bot. If it scores above 0.60, mine its wins for patches (vast #8).
5. **Opponent archetype classifier**: gate it with a payoff matrix of 4 param sets × 6 archetypes first (vast #5, strategy #6).
6. **Learned launch/commit value** from replays (P(win | launch) under fog), AUC gate ≥ 0.72 (vast #6).
7. **Castle-build strike window**: strike while the enemy just paid for a castle near its general (strategy #2).
8. **Race calculator**: our ETA to kill vs theirs before choosing defend or attack (strategy #1; partly covered by `try_kill` first).
9. **Deathtouch endgame solver**: only if ≥ 5–10% of our games reach turn 800.

## Holdout set (never tune on)
nanomena-BC and Kubic-BC (when trained), ResBot-BC at temperature 0, ext_superbot, ext_sentinel, zoo_mixed,
and a frozen early version (v1).

## Experiment log (Ada, 120 games per opponent unless noted; pooled score ± 95% CI)
| Change | Result | Decision |
|---|---|---|
| early expansion to T100 (`early_expand_until=100, bonus=6`) | 0.738 → 0.766; t1c 0.57 → 0.68 | **adopted** |
| ring defence (`ring_w=0.7, r=2`) | 0.738 → 0.738 | not adopted (tunable) |
| castle enemy-facing pref (`castle_front_w=1`) | 0.738 → 0.742 | not adopted (tunable) |
| stealth routing 0.5 / 1.5 / 3.0 | 0.782 / 0.726 / 0.716 (base 0.738) | **adopted 0.5** |
| lead-conditioned aggression 0.5 / 1.0 | 0.723 / 0.703 | rejected |
| chase-kill interceptor (algo-study 2.1) | 0.738 → 0.753 | **adopted** |
| general-drain veto | identical (never triggers) | dropped |
| capture-rate leaks | 0.741 (t1c +16, bc −8) | tunable flags (default off) |
| castle gating | 0.718 | rejected |
| belief land bound | identical | dropped |
| learned threat GBM (local) | no gain | off, tunable |
| EklipZ fog queue (offline check) | worse than 0.3×hidden bound | off |
| early castles (local, 30 games) | 0.567 → 0.583 (n.s.) | left to the tuner |
| candidate params (7 opponents) | c_base 0.772, c_base_es 0.811, **c_a2best_es 0.829 (bc 0.79)** | ada2 params + es is the best so far |
| attack-only-when-ahead gate (ratio 1.0 / 1.2) | 0.561 → 0.416 / 0.086 vs strong pool | **rejected** (launches are what win) |
| castle-from-general / stack castles (castle_g, stack price) | 0.626 → 0.561; with early castles 0.26–0.44 | **rejected** (defaults off); castles stay late (tuned) |
| kill-front gathering when the enemy general is known | 0.425 → 0.356 vs big clones (80 games, local) | **rejected** (default off; tunable) |
| embedded threat GBM | no gain + audit risk (replay-trained, 141 KB) | **removed from the bot** |
| fresh-map overfitting check (100 games/opp, maps never used in tuning) | c_a2es3 0.670 fresh vs 0.653 tuning maps; ada5c 0.672 / 0.669; local7b 0.650; t1c 0.649 | **no map overfitting**; tuned candidates ≈ c_a2es3 |
| es2 (ES over modulator) | started on tune_base9 = default params (incumbent 0.10 vs c_a2es3) | **cancelled**, restarted as es3 on tune_base10 = participant + a2es3 params |
| diagnosis vs ResBot-BC (24 games) | T100 land 42 vs 53, T300 land 57 vs 92, castles 0 vs 3.5; their attacks on our land T50–200: 22 vs our 13 | economy gap = land war + castles |
| castle start 120 / 160 vs clones (80 paired games, local) | 0.425 → 0.31 / 0.31 | **rejected**: our castles get drained to 1 and captured (7/8 lost in traces) |
| castle_keep (half drains; none near enemy), local 120 games/variant | 0.529 → 0.483; + early castles 0.471; + stack/ring castles 0.367 | **rejected** (replays: top bots drain castles fully too) |
| strategy-port on Ada vs 4 clones (80 games/opp) | base 0.352, corridor 0.336, **combo (sweep+corridor) 0.400**, **stage (sweep+stage) 0.384** | sweep merged as flag; re-tested on current base (sw1) |
| x_early_enemy (enemy captures get the early-expansion bonus) | 0.549 → 0.569 (n.s.) | not adopted |
| x_kill_stealth (stealth path for kill runs > 8) | 0.543 → 0.546 | rejected (no effect) |
| vs hvn (laptop, 40 games) | base 0.50, c_a2es3 0.50, t1c 0.625; hvn land 82–123 vs our 57–62 (T150–400), no castles | our land plateaus after T150 → x_spread / x_phase_enemy tests |
| x_phase_enemy (enemy captures x2 at turn%50 >= 35, x0.3 below 15), 3 settings | 0.541 → 0.467 / 0.486 / 0.444 | **rejected** |
| x_spread (post-bonus small-stack neutral spreading) | clones+: 0.555 → 0.458; hvn 60 games 0.667 → 0.583 | **rejected** |
| **x_sweep** (exact path walk in the kill check, 10 candidate stacks) on current base | 0.580 → **0.620**; + x_stage 0.585 | **adopted** (default on, tune_base12) |
| es3 → es4 | es3 (base10) superseded after 3 gens; es4 = ES over m_* + sweep_cands on tune_base12, warm start from es3 mean_avg | running (5 h) |
| deathtouch guard (dt_guard: chase / kill / reinforce the threatened neighbour from T790) | fixes the T865 loss vs hvn (we were ahead 1253 vs 895); unit tests: block fails without it | **adopted** (x_dt_guard=1) |
| win rate vs ResBot-BC by general BFS distance (120 games) | 16–23: 0.35–0.39; 24–35: 0.17–0.29 (small n) | noted; no restart of es4 |
| **before/after** (fresh maps offset 6000, 100 games × 10 opponents) | c_a2es3 0.678 → **tune_base12g 0.712** (bc_resbot128 .37→.42, nanomena .475→.515, rusher .78→.88, kubic .47→.45) | current base adopted |

Note: fresh-map indices wrap modulo 2000 (arena/run.py), so offsets 4000/6000 reused maps 0–49 (same as fresh1);
comparisons inside one eval are still paired. Decision evals used fresh maps 0–49, 400–449, 500–549, 1000–1049,
1500–1549. **Final selection uses offset 1700 (maps 1700–1749), untouched by any decision.**
| before/after vs public bots (laptop, 40 games each) | c_a2es3 0.743 → tune_base12g 0.779 (sentinel10 .75→.90, hvn .65→.75, bca .525 both) | confirms |
| kill_margin 0 / 5 / 9 under x_sweep (vs tuned 2) | 0.551 / 0.528 / 0.536 vs ≈0.566 | keep 2 |
| early_expand_until 130 / 160 (bonus 7) | 0.645 → 0.615 / 0.557; vs hvn+bca 0.74 → 0.66 (160) | **rejected** |
| bug hunt: even-army losses vs t1c/hvn/sentinel/boss | (1) launch stack kept marching away while a 91 stack closed in (garrison non-urgent until eta 5) → x_def_escalate; (2) two enemy stacks merged next to the general (need = max over single threats) — position already lost, no fix | |
| x_def_escalate (urgent garrison if deficit ≥ 25% of need and eta ≤ 20) | Ada 540 paired: 0.683 → 0.699 (de3 0.696, de2 0.686); public bots 120 g: 0.892 → 0.879 | neutral; candidate in final selection |
| slot asymmetry vs clones (80 games each) | resbot96 P0 0.50 / P1 0.35; kubic P0 0.50 / P1 0.375; t1c P0 0.675 / P1 0.70 | clone-side asymmetry (not our bot: symmetric vs t1c) |

## Final selection (fresh maps 1700–1749, never used for decisions; 100 games × 10 opponents)
| bot | pooled | resbot96 | resbot128 | nanomena | kubic | t1c | sniper | flash | mixed | rusher | hunter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c_a2es3 (morning best) | 0.665 | .36 | .32 | .48 | .41 | .72 | .78 | .92 | .87 | .80 | .99 |
| F0 = base (sweep + dt_guard) | 0.7125 | .45 | .39 | .57 | .445 | .69 | .89 | .94 | .925 | .84 | .985 |
| F1 = F0 + escalation | 0.7075 | .41 | .37 | .52 | .445 | .71 | .93 | .945 | .93 | .83 | .985 |
| **F2 = F0 + es4 modulator** | **0.752** | **.555** | **.58** | **.66** | **.66** | .57 | .92 | .90 | .94 | .775 | .96 |
| F3 = F2 + escalation | 0.746 | .565 | .57 | .64 | .64 | .55 | .92 | .90 | .94 | .775 | .96 |

Public bots (laptop, 40 games each, maps 10000+): F0 0.816 (sentinel10 .825, hvn .675, juraj35 .887, superbot .925, boss .95, sentinel .825, bca .625);
F2 0.821 (.85, .775, .875, 1.0, .85, .95, **bca .45**).
| F2h (es4 modulator × 0.5), maps 1700 | 0.7245 (clones .505/.44/.565/.58, t1c .65, hunter .99) | interpolates; no free lunch |
| **confirmation slice 1800–1849** | F0 0.692 vs **F2 0.754** (clones .64/.545/.69/.64 vs .46/.34/.46/.54; t1c .57 vs .675; hunter .90 vs .96) | F2 gain replicates; weaker vs t1c/hunter |
| es4 modulator, what it learned | garrison bias −0.48 (+0.45·army ratio), launch bias +0.51 (+0.26·enemy castles), kill bias −0.24, capture bias −0.35 | mostly global aggression → test F2 minus garrison weights (F2g) |
| F2g / F2gb (es4 minus garrison weights / bias), maps 1700 | 0.739 / 0.7295 (F2g: clones .47/.51/.69/.49, t1c .64, hunter .97) | more robust, less strong |
| bca + boss, 60 more games (total 100) | bca F0 0.53 vs F2 0.435; boss 0.93 vs 0.90 | |
| **DECISION** | **F2** = participant + es4 mean_avg: best pooled on both untouched slices (0.752, 0.754 vs 0.71, 0.69) and best worst case (0.435 vs ~0.34–0.39) | submission params = F2 |
| gate on F2 (20 bots × 30 games) | 0 forfeits / 600; worst move 36 ms under load; min zoo_gatherer 0.833; avg 0.953 (F0 0.944) | pass (no catastrophes) |
| subprocess timing F2 (1 core, 150 ms, 3× slowdown) | worst move 56.6 ms, worst first move 529 ms, 0 timeouts | pass |
| built file `submission/participant_id.py` (placeholder ID) | 92,701 bytes, PARAMS == F2, DEBUG False, sha256 32256f05…74ee; 80-game sanity 0.756 | rebuild with the real ID/name |

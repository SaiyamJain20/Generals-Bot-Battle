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

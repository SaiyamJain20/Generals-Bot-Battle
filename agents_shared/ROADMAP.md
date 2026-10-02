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

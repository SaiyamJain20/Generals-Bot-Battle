# vast-research: broad game-AI sweep, mapped to our bot (2026-10-02)

Scope: web research + reading only. Nothing run locally. Read first so as not to repeat: strategy-research.md
(local 2-ply search, deathtouch solver, race calc, castle strike window, fog heat map, opp profiles),
algo-study.md (EklipZ/relh/juraj ports: chase-kill, fog tracks, chokes, capture-rate), training-research.md
(league, PFSP, SPRT/Elo CIs, holdout, CMA noise), replay-analyst.md. Where an idea overlaps, I cross-reference
(`[SR#]`, `[AS§]`, `[TR]`) and only add what is new.

Evidence labels: [V] = read on the fetched page this session; [S] = seen only in a search snippet;
[U] = my background knowledge, UNVERIFIED here (treat as hypothesis).

Test conventions used below
- Arena: `.venv312/bin/python tools/abpool.py BOT GAMES WORKERS 'specJSON' ... --opps a.py b.py ...` (max 3 workers while sharing CPU).
  abpool uses `map_offset=600`, so every spec sees the SAME maps (paired by map; side-swap is inside run_match [U: check]).
- Noise: at p~0.5, SE = 0.5/sqrt(N): N=200 -> +-3.5 pts (95% CI +-6.9); N=400 -> +-4.9 CI; N=1000 -> +-3.1 CI.
  Do not accept a change on one pool of <200 games unless the effect is >7 pts.
- POOL_STRONG = bots/opp/ext_sentinel.py ext_juraj35.py ext_hvn.py ext_amin.py (+ ext_bca.py with 2 workers, 64 ms, 550 MB).
- POOL_SIMPLE = hunter.py rusher.py turtle.py expander.py zoo_flash.py zoo_castler.py zoo_gatherer.py zoo_sniper.py zoo_turtle_dt.py zoo_mixed.py.
- Holdout = opponents NOT used in tuning (e.g. ext_humanexe, ext_superbot, ext_doomstack, ext_mybot9, the BC clone) and a different map_offset
  (copy abpool to a scratch script with a new offset; do not edit coordinator tools).
- "No-regret rule": a change is adopted only if pooled score on POOL_STRONG+POOL_SIMPLE rises AND no single opponent drops by > 8 pts AND holdout does not drop.

## 0. Three framing insights (these change how the ideas rank)

1. A fixed mixture of strategies can NEVER beat the best pure strategy against a FIXED opponent field: payoff is linear in the mixing
   weights, so E[mix] = sum w_k E[k] <= max_k E[k]. [U, elementary]. Randomising between our parameter sets only helps (a) against an
   ADAPTIVE opponent that learns our style, or (b) in the minimax sense (worst-case over a field we do not know). In a round robin of fixed bots,
   the real expected gain comes from CONDITIONING on the opponent (idea 4), not from mixing (idea 1). Also the field in a knockout is
   NOT the field of the round robin: survivors are the strong bots, so tuning-pool weights should tilt to strong opponents (idea 2).
2. The strongest generals.io agent found, AverageJoe / "Superhuman AI for Generals.io" (arXiv 2606.23348), is a 15M-param ViT trained with
   sparse win/loss reward, advantage-filtering and parameter-EMA; it beats Human.exe 100%, top humans ~64-74%. [V] Its emergent behaviour
   (feints, backdoor stacks hidden in fog, snowballing from a small lead, attention-head belief over general location) is the template
   of what "good" looks like, and each of those has a cheap heuristic analogue (ideas 5, 6, 7).
3. Contest winners repeatedly report: a rigorous arena with confidence intervals beat clever algorithms (Agade, Ghost in the Cell: "a new
   version would beat the previous one, even by 0.5%, or be thrown away") [V]; and over-engineering opponent modelling HURT (pb4, Fall
   Challenge 2020: simple opponent model +5 Elo, complex one -10 Elo; wider beam -15 Elo; search time capped at 50% cost 0 Elo) [V].
   => Prefer cheap, testable, conditionally-triggered features; budget extra search only where contact happens.

## 1. RANKED IDEA LIST (25 ideas)
Columns: impact (expected pts of win-rate vs a mixed field, my guess) / effort / certainty. "T" = needs test plan (section 2).

| # | Idea | Where in code | Impact | Effort | Cert. |
|---|------|---------------|--------|--------|-------|
| 1 | Stealth routing: path cost for cells inside enemy vision (3x3 of enemy-owned land), arrive "backdoor" (feint/backdoor of AverageJoe) | `path_to`, `choose_target`, `cycle_move` launch path; new PARAMS `stealth_w`, `stealth_until_d` | +2..6 vs reactive bots, 0 vs maphack-ish RL | S | T |
| 2 | Pool/fitness reweighting for knockout: strong-heavy weights, softmin (smooth worst-case) term, paired maps + side swap, holdout gate | `tune/cma_tune.py`, abpool | +1..3 true strength (reduces overfit) | S | T |
| 3 | CMA output = averaged mean of last K generations (EMA analogue, +30 Elo in AverageJoe) instead of best sample; re-evaluated top-3 | tune post-processing only | +1..2 | S | T |
| 4 | Opponent-archetype classifier by t~80-150 -> small param offsets (garrison, w_launch, castle_start) | new `classify_opp()` in `update`, offsets applied in `decide` | +2..5 IF payoff gap exists | M | T (gate test first) |
| 5 | Learned launch/commit value V(win\|launch state) (Texel-style logistic on 15k replays) gates `try_kill`, all-in, cycle launch | `cycle_move` launch, `try_kill`, `endgame`; embed 15-25 coefficients | +2..4 | M | T |
| 6 | Imitation priors: label which of OUR option types the top bot's actual move matches; learn option score weights (replace/regularise w_*) | `macro`, `decide` weights `w_garrison,w_build,w_cycle,w_launch,w_scout,w_home_fill` | +1..4 | M | T |
| 7 | Lead-conditioned aggression (snowball): multiply w_launch/v_enemy by a sigmoid of land+army lead; mirror when behind | `macro`, 4-6 new params; CMA tunes | +1..3 | S | T |
| 8 | Catastrophe suite + slow-CPU stress + hard wall-clock fail-safe returning a safe move | `act`, `_decide`, tools/dt_stress.py | prevents -forfeits | S | T |
| 9 | Contact-triggered quiescence: when enemy stack within r of our stack/general, extend exact sim 1-2 ply vs a SMALL reply set (continue, toward-general, hold) taking min | `resolve` exists; new `contact_search()` | +2..4 [SR1 overlap] | M | T |
| 10 | SPRT stopping in abpool/ab4 (fishtest GSPRT): stop clear wins/losses early, 2-3x fewer games | tools only | throughput | S | no |
| 11 | Exploiter loop: tune a clone ONLY against frozen current bot; if >0.60, mine its loss games and patch | tune + lossscan | finds holes | M | T |
| 12 | Meta-game payoff matrix over K CMA sets x pool; Nash LP used for POOL WEIGHTS and as diagnostic (non-transitivity), not necessarily for deployment | offline script | diag | S | T |
| 13 | Deploy a mixed set (seeded by map hash) ONLY if matrix shows cycles (rock-paper-scissors) with minimax gain >= 3 pts | `PARAMS` selection at `Bot.__init__` | 0..+2 | S | T (gated by 12) |
| 14 | Retrograde/tablebase for deathtouch endgame: precompute table (dist, my army bucket, enemy army bucket, who moves) -> win/lose/draw via exact engine; embed < 20 KB | `endgame`, `try_kill` [SR2 overlap] | +1..3 on late games | M | T |
| 15 | Opening book by map class: decision tree from (gen-corner openness, wall density, dist to nearest candidate enemy site) -> `open_cfg`, offline from 20k maps | `plan_opening` (add cfgs, prior from tree) | +0..1.5 (replay-analyst: opening not discriminating) | M | T |
| 16 | Draw avoidance: when ahead at t>=600 raise aggression/all-in; when opponent is a turtle (profile) start siege earlier | `endgame`, `macro` | + on turtle matchups | S | T |
| 17 | Anti-rush early garrison calibrated vs rusher/flash zoo (first 150 turns) | `garrison_need`, params `garrison_*` | prevents loss streaks | S | T |
| 18 | Information denial: avoid general moves, avoid exposing castles/stacks in enemy vision near home; do not build castles where price pattern reveals cluster | `gen_move_pen`, `castle_site` | +0..1.5 | S | T |
| 19 | Scoreboard belief: use opp_land/opp_army history (AverageJoe feeds 512-step history tokens) to update general-location belief and mass estimates | `belief_scores`, `update` [AS1.1,1.8 overlap] | +1..2 | M | T |
| 20 | Move ordering + anytime evaluation: order candidates by prior score, stop at budget (iterative deepening analogue); keep per-turn BFS/dist caches (terrain static) | `decide`, `bfs`, `enemy_distance_map` | frees ms for 9 | S | no |
| 21 | Simultaneous-move local matrix solve: at a contact point build the k x k payoff of (my option x enemy option) by exact `resolve` and play the maximin (pure or tiny mix) | `contact_search` | +0..2 | M | T |
| 22 | Shaped-fitness variance reduction for CMA (blend win with tanh(land-lead), keep sparse for final selection) | `tune/cma_tune.py` | tuning speed | S | T |
| 23 | Variance control in series: if opponent classed stronger by t~100, switch to a pre-tuned "high-variance" set (more castles/all-in), if weaker use "low-variance" safe set | param sets | speculative | M | T (low priority) |
| 24 | Texel fitting of hand constants (v_enemy, v_ecastle, bonus_mult, castle value): logistic regression of outcome on features at sampled replay states, use as CMA initial mean | offline | better init | M | T |
| 25 | PBT-style population: keep 6-8 diverse tuned sets (rectified-Nash selection of who to train against) | tune | pool quality | L | later |

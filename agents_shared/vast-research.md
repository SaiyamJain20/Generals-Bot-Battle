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

## 2. WHAT/WHY AND TEST PLANS (uncertain ideas)

Common: BASE = current best (bots/participant.py or newest bots/versions/*). A/B = same file with PARAMS overridden via the JSON spec, so
a parameter-only idea needs no code fork. Code ideas: copy to `bots/versions/vr_<idea>.py` (our own scratch file) with the new params default-off.

### Idea 1. Stealth routing
Mechanism: vision is 3x3 around owned cells, so an attack stack that walks along cells adjacent to enemy land is seen 1-2 turns before
contact; reactive bots (relh/Sentinel tracks, hvn, juraj approach FSM [AS§2]) gather defenders. AverageJoe's "backdoor" and "feint"
behaviours are emergent [V]. Heuristic: in `path_to` add cost `stealth_w` for cell j if any neighbour (8-nbhd) is enemy-owned-and-known
(or in the enemy's last-seen footprint), except for the last `stealth_until_d` cells before the target (final approach cannot hide).
Plus prefer a path whose predecessor cells have been fog for us (unknown = cheap) -- optional.
Test: specs `{}`, `{"stealth_w":0.5}`, `{"stealth_w":1.5}`, `{"stealth_w":3}` (x `stealth_until_d` in {2,4}) = 8 specs.
 Round 1 (screen): 120 games/spec x POOL_STRONG(4) = 480 games each?? too many -> use 60 games per opp x 4 = 240/spec, 3 workers.
 Metrics: pooled score; per-opponent; mean length of games won; "time from first enemy sighting of our stack to contact" (log in trace).
 Decision: carry the best 2 specs to round 2 with 200 games/opp on STRONG + 100/opp on SIMPLE; adopt if pooled +3 pts (CI excludes 0) and no-regret rule.
 Kill rule: if stealth_w=1.5 loses > 4 pts pooled (detours cost tempo) drop. Risk: longer paths = later attacks; bound detour length <= 4 extra steps.

### Idea 2. Reweighted pool/fitness (knockout field)
Reason: knockouts select for strong opponents; a pool dominated by weak bots rewards the wrong optimum. Also CMA's best sample is noisy [TR].
Test (offline, cheap, uses existing result tables): from the 20 most recent CMA candidate evaluations, recompute rank under (a) uniform pool
mean, (b) 70% strong / 30% simple, (c) softmin(tau=0.05) over per-opponent scores. If rank correlation (Spearman) between (a) and (b) < 0.7 the
objective matters -> then run one tune with (b)+(c) and compare winners on HOLDOUT with 300 games per opp. Adopt if holdout >= +2 pts.

### Idea 3. Averaged CMA mean (EMA analogue)
Evidence: parameter-EMA beat last iterate by ~30 Elo in AverageJoe after 6 days [V] (different setting: SGD noise; CMA mean is already a
recombination, so the gain may be smaller [U]).
Test: take the final run's logs: candidates X_best (best sample), M_last (distribution mean), M_ema (mean averaged over last 5-10 generations).
 Evaluate all 3 on fresh maps (offset not used in tuning), 300 games x POOL_STRONG(4)+ HOLDOUT(3). Decision: choose the highest; if M_* beats
 X_best by >= 2 pts adopt as standard post-processing; if within +-2 pts use M_ema (lower variance) anyway. Note CMA params are continuous; integer
 params round after averaging.

### Idea 4. Opponent archetype -> parameter offsets (GATE TEST FIRST)
Why: this is the only form of "opponent modelling" with a positive expected gain against fixed bots (insight 0.1). Fast adaptation within ONE
game (~450 turns): signals are available by t=80-150: first-contact turn, enemy land growth rate, enemy army/land ratio (hoarder vs expander),
first castle turn, whether enemy stack>=0.25 army moves toward us before t=150 (rusher). Keep to <=4 archetypes; pb4 found simple models good,
complex ones harmful [V].
GATE test (no bot code): build matrix M[set][archetype] where sets = {BASE, BASE+defensive offset, BASE+aggressive offset, BASE+castle-heavy} (specs)
 and archetypes = {rusher: rusher.py+zoo_flash.py; turtle: turtle.py+zoo_turtle_dt.py; expander: expander.py+zoo_expander_plus.py; gatherer/sniper
 zoo; strong-RL: ext_amin+ext_bca; strong-heuristic: ext_sentinel+ext_juraj35+ext_hvn}. 100 games per cell (about 4 x 6 x 100 = 2400 games).
 Decision: classifier is worth building only if, for >= 3 archetypes, the best set beats BASE by >= 5 pts AND best sets differ across
 archetypes. Else drop (BASE is already robust). If it passes, implement classifier from features above; evaluate classification accuracy
 on logged games (target >= 80% by t=150) before A/B of the full switch (200 games/opp over all pools; adopt at +2 pts pooled).
Consider offsets applied smoothly (posterior-weighted) rather than hard switch, to avoid thrash.

### Idea 5. Learned launch/commit value
Data: 15k full-information replays of top bots (we know both sides). Sample states at each launch event (stack>=30, moving toward the enemy;
replay-analyst already extracts strikes) -> label = launcher wins the game (or: captures the general within 60 turns). Features: stack/enemy-gen-army
ratio, path length, our/their land+army ratio, turn, castles, belief entropy (what we would know under fog, NOT privileged info), distance of
enemy visible stacks. Fit logistic (and compare with depth-3 GBT). Deploy: gate cycle launch and all-in on V > threshold; replace fixed
`attack_min_army`.
Test: (a) offline: AUC and Brier on held-out replays by bot (train on 12k, test on 3k) -- need AUC >= 0.72, calibration slope 0.8-1.2.
 Use ONLY fog-visible features (re-mask replays through the engine's observation) else leakage is the usual failure. (b) Arena: spec
 `{"commit_w":0}` vs `{"commit_w":1}` x thresholds {0.4,0.5,0.6}; 200 games/opp STRONG+SIMPLE. Adopt at +3 pts pooled.
 Texel-style (fit eval to outcomes) is the engine-world analogue [U]; the "tuning evaluation by logistic regression on game results" is standard.

### Idea 6. Imitation of the top bot to set option priorities
Method: replay the top bot's games (full obs) through OUR option generators (macro options and their proposed moves); label each ply with the
option whose proposed move equals the actual move (ties -> mark ambiguous, drop); fit multinomial logistic on cheap features (turn bucket, our/their
army ratio, threat flag, castle affordability, frontier size, stack count>=min_stack). Output: priors p(option|state). Use as additive
log-prior `imit_w * log p` on option scores, with imit_w tuned by CMA (so it can go to 0).
Test: (a) top-1 agreement of the learned chooser vs the current weights on held-out replays (current weights are the baseline). If agreement gain < 5 pts
 the heuristic already matches, stop. (b) A/B `imit_w` in {0, 0.3, 0.6, 1.0}, 200 games/opp STRONG+SIMPLE; adopt at +2.5 pts pooled and no regressions.
 Caution: imitating a stronger bot's choices without its eval can desynchronise (copying action types in states we reach differently); hence the
 tuned scale. Distilling the BC neural clone into a tree is the same pipeline with clone labels (unlimited data, noise-free) -- prefer clone labels for
 off-replay-states (use DAgger-like: run our bot, ask the clone what it would do, label disagreements).

### Idea 7. Lead-conditioned aggression
AverageJoe: "once the agent has even a small lead, it balances expanding and attacking, slowly pushing the advantage until it wins" [V]; relh's
land-deficit modes [AS§2.7]. Add `lead = tanh((land_me-land_opp)/k1) + tanh((army_me-army_opp)/k2)` known from the scoreboard (always visible),
and params `lead_launch_gain`, `lead_garrison_gain` (+-). Test: 3 params gain in {0, 0.3, 0.6} grid (9 specs is too many; coordinate sweep: 5 specs).
 200 games/opp STRONG + 100/opp SIMPLE. Adopt at +2 pts. Compare also draw rate at t=1200 (should drop).

### Idea 8. Catastrophe suite
Make an always-run gate before any submission: (i) simple bots must be >= 0.95 each (pass_bot, random_bot, rusher, turtle, hunter, zoo_*), 40 games
each; (ii) zero exceptions/timeouts in 300 games (count forfeits); (iii) slow-CPU stress: run with `taskset -c 0` on a loaded core with 3 competing
busy loops (emulates 2-3x slower grader) and check max move time < 120 ms; use tools/dt_stress.py [exists]; (iv) wall-clock fail-safe: if elapsed > 110 ms
mid-decision return best-so-far/ safe move (iterative-deepening style) -- verify `soft_budget_ms` really bounds the worst case (log p99.9 and max per turn
over late-game turns with many stacks). Decision: any (i)-(iii) violation blocks submission.

### Idea 9 / 21. Contact quiescence and local matrix solve
Chess "quiescence" = do not evaluate in the middle of a capture sequence. Analogue: our exact 1-turn sim evaluates a position where an enemy
stack is adjacent to one of ours, but the outcome is decided in the NEXT exchange. Trigger only when (enemy-visible stack within 2 of our stack>=X
or within 4 of our general); then build a small enemy reply set R = {continue last enemy move, move toward our general, attack our nearest stack,
hold} and compute min over R of the 2-ply eval via `resolve`. For simultaneous-move games the principled version is the matrix-game solve over
(my k options x |R|); take the maximin pure option (solve the mixed LP only if the matrix has no saddle; fixed priority order breaks symmetry, so
saddle points are common [U]). Time: only on contact turns; budget <= 25 ms. [SR§2.1 has the 2-ply idea; this adds the trigger and the reply set.]
Test: log the fraction of turns triggered (target 5-15%), p99 time. A/B `{"contact_ply":0}` vs `{"contact_ply":2}` 200 games/opp STRONG+SIMPLE; adopt +2.
 Targeted test: construct 300 contact positions from replay snapshots (stack adjacent to enemy stack, near general) and measure whether the
 chosen move survives vs the real next enemy move (top bot's actual) -- "tactical solve rate" before/after.

### Idea 11. Exploiter
AlphaStar: main exploiters play only the current main agent to find holes; league exploiters find systemic weaknesses [V search snippets].
Plan (cheap version, no RL): CMA-ES over a copy of our parameter vector, fitness = score vs FROZEN current best only (60 games/eval x ~500 evals
 ~1h on 3 cores [estimate]). Check: exploiter vs frozen on fresh maps, 300 games. If > 0.60 we have an exploitable hole -> use lossscan.py/trace.py
 on the exploiter's wins to find the pattern (e.g. early rush at distance X, castle denial), patch it, and add the exploiter to the pool.
 If <= 0.55, the parametrisation space is exhausted (good news). GPU version: fine-tune the BC clone with RL vs our frozen bot (heavy; only if the cheap
 version found something). Decision recorded in OPEN_QUESTIONS Q10.

### Idea 12/13. Meta-game payoff matrix, Nash, PSRO
Procedure (offline script, scipy allowed): K=6-10 candidate sets (CMA winners from different runs/pools + defensive/aggressive/castle-heavy variants)
 x R=8-12 opponents (pool incl. the K sets themselves for K x K). Cells 100 games each (K x (K+R) x 100 ~ 15k games; do in batches, reuse
 existing evaluation logs). Solve: (a) max-min mixed strategy over the K x K block (symmetric zero-sum: value 0.5); (b) max over mixtures of the
 WORST opponent-in-pool score; (c) best pure. PSRO view: the sets are the oracle's policy population, Nash of the restricted game is the
 meta-strategy, new best-response candidate = CMA tune against that mixture; "rectified Nash" restricts best responses to opponents that the learner
 beats/ties to keep diversity [V snippet] -- practical use: choose which opponents the NEXT tune plays against.
 Decision rule for DEPLOYING a mixture (idea 13): bootstrap the matrix (resample games); deploy only if the lower 90% bound of
 [minimax value of mixture - best pure minimax value] >= +2 pts AND the mixture's average-field score is within 1 pt of best pure. Otherwise ship the
 best pure (insight 0.1). The matrix is useful anyway to detect intransitive cycles (A beats B beats C beats A), which is the case in which
 pure strategies are unsafe in a knockout.
 Implementation of mixing is trivial: choose index by hash of the initial observation/map and player id, constant for the whole game
 (commitment mid-game is better than switching, since our parameters interact with state such as castle plans).

### Idea 14. Deathtouch endgame table
We have a pure-Python exact engine, so a retrograde table over a small abstraction is feasible: state = (distance between the two decisive
stacks / generals in {1..12}, our strongest stack bucket (log scale 12), enemy strongest bucket (12), tempo parity {2}) -> {win, loss, draw} /
expected capture timing, computed by value-iteration on the abstract rules derived from engine (deathtouch rule from engine source). Size
12 x 12 x 12 x 2 = 3.5k entries (< 10 KB embedded). Use in `endgame` to decide "commit all-in vs hold/garrison" at t >= 700.
Test: (a) fidelity: sample 500 real endgame snapshots (replays t>=800), compare table's verdict with exact engine rollouts using our bot on both sides
 (agreement target >= 80%); (b) arena from snapshots: start 300 games from late-game snapshots (if the arena can load states; otherwise from a
 forced-long game vs turtle_dt/hunter to reach 800+), `{"tbl":0}` vs `{"tbl":1}`; adopt +3 pts on those.  Q9 in questions.md (share of games reaching 800)
 decides whether any of this matters: if < 5% of games reach 800, downgrade to "do not build".

### Idea 15. Opening book
Replay-analyst found openings not discriminating among strong bots; our `plan_opening` already simulates 32 configs per game in <= 1.5 s [code].
Cheap test first: tools/opening_eval.py: compare land at t=50 (and t=100) of the online-planned config vs the best single static config vs an
 oracle (best of a 200-config sweep per map, offline). If online is within 1 land of the oracle on >= 90% of maps, stop. Otherwise train a tree (features:
 free-neighbour count, walls in radius 6, corner/center, distance to nearest candidate site) to propose configs; add them to the online search.
 Arena check only if land50 improves >= 1.5 average: 200 games/opp STRONG+SIMPLE; adopt at +2.

### Ideas 16/17. Draw avoidance and anti-rush
Metrics from a standard 100-game run per opponent: draw rate at t=1200 (target <= 3% vs everything except mirror clones), loss rate before t=200 vs
 rusher/zoo_flash (target <= 2%), median game length. Offline: run `{"garrison_min": 2,3,4}`, `{"garrison_frac_hidden": 0.3,0.5,0.7}` on
 rusher+zoo_flash+zoo_sniper 100 games each; pick the Pareto point (loss-rate on rushers vs score vs gatherers/turtles). Draw rule: if our
 land+army lead >= 15% at t=600 spec `{"push_t":600,"push_gain":x}`.

### Idea 18. Information denial
What the opponent can read: (1) visible cells in its 3x3 vision, (2) scoreboard (our land/army totals each turn: the army DROP when we pay for
a castle plus the price formula 35+sum max(0,14-2d) leaks the build distance structure -- our own `register_enemy_build`/`locate_new_castle`
exploits the same leak on the enemy [code]), (3) general position by leaving it moving. Cheap defences: build castles on turns when other army moves
also change the total (a build coinciding with big captures hides the price), do not move the general out of fog-protected interior, keep the
home perimeter wide (cells >= 2 deep) so a scout cannot see the general tile. DeepNash: bluffing/hiding emerges when information is valuable
(positive/negative bluffing) [V snippet]; but vs scripted bots it is mostly irrelevant, vs RL clones it may matter.
Test (cheap, diagnostic): measure how often OUR bot (`locate_new_castle`) correctly localises enemy castles from price. If > 60% accuracy, the leak is
real; then test the defence only against a version of the opponents given the same inference (we can code a spy wrapper on a zoo bot) --
only worth it if leak accuracy is high. Otherwise skip.

### Ideas 22/24. Fitness shaping and Texel init
Sparse win/loss reward converged cleanly at scale; shaped reward destabilised late for RL [V]. For CMA with 60-200 games per evaluation the opposite
trade-off holds: the binary signal has SE 3.5-6 pts, so a low-variance proxy helps early generations. Test: on existing logged evaluations, compute
correlation between win-rate and (mean final land-share, mean captured-general turn) across candidates; if corr >= 0.6 use 0.8*win + 0.2*proxy for the first
half of generations and pure win-rate afterwards; adopt if the resulting best is >= 1 pt better on HOLDOUT with the same game budget (needs 2 runs; so only if CPU is free).

### Idea 23. Variance control in series
In single-elimination vs a stronger opponent variance helps; vs weaker, hurts. Without knowing bracket/format details we cannot exploit this reliably;
test only after idea 4 gate: need a "high-variance" set whose score vs STRONG is lower on average but with more extreme games, and a measurable
advantage in P(win>=1 of 2 games). Compute P(at least one win in 2 games) = 1-(1-p)^2 for each set vs STRONG; if a set has lower p but higher
P(1-of-2) the format favours it. [U] Lowest priority.

## 3. WHAT EACH FIELD TEACHES (and what we already have / skip)

### 3.1 Game theory (simultaneous, imperfect information)
- Mixed strategy matters only when opponents adapt or when you face a worst case; against stationary scripted bots maximise expected payoff
  against the (strength-weighted) field instead (insight 0.1). CFR/regret-matching: needs repeated play with counterfactual values; a single 450-turn
  game with a sparse end result has no per-decision counterfactual we can evaluate in 45 ms, so do NOT embed CFR. Use regret-matching / multiplicative
  weights OFFLINE only, as a solver for the K-set payoff matrix (equivalent to LP, simpler in stdlib) [U].
- Double oracle / PSRO: population of policies + restricted-game Nash + best response; our CMA tune against a pool is the oracle; the Nash of
  the restricted game gives the pool distribution [V snippets: PSRO surveys, Pipeline PSRO, Rectified Nash]. Implementation = idea 12.
- DeepNash (R-NaD) reached top-3 all-time on Gravon by regularised self-play, learning positive/negative bluffing [V snippet]; irrelevant as an
  algorithm for us (RL scale), relevant as a reminder that information has value.
- Round robin of 2 games per pair: if sides are swapped, deterministic play gives the same outcome type; nothing to randomise. If the opponent is
  one of the top RL bots they are deterministic too (argmax) [U]. Fixed priority resolution is a known tie-breaker, not exploitable by us alone.
- Opponent modelling speed: pb4 saw +5 Elo for a minimal opponent model (mark what the opponent will likely take) and -10 Elo for a full opponent
  simulation [V]; Agade predicted opponent bomb targets by self-playing 10 turns with no bombs [V]. => Predict with our own policy as the proxy
  (cheap "what would I do in their place" = mirror model) rather than learning a model; we already have `enemy_general_threats`; extend with mirror
  prediction of enemy's next launch target (low effort, part of idea 9's reply set).

### 3.2 Chess/Go/shogi engine techniques
- Texel tuning (logistic regression of eval to results), SPSA, fishtest SPRT [U; standard]. Ours: ideas 5, 24 (Texel), 10 (SPRT), CMA-ES already
  covers SPSA's role. SPRT formula for W/L only: LLR += ln(p1/p0) per win, ln((1-p1)/(1-p0)) per loss; p0 = 0.5 + elo0-ish, bounds ln(b/(1-a)), ln((1-b)/a), a=b=0.05.
  For "is B better than A by +3 pts", H0: p=0.5 vs H1: p=0.53 needs ~ 1.5k games on average; so use SPRT for GO/NO-GO of obvious (>= +6) and
  fixed-size for subtle ones. Use `elo0=0, elo1=+20 (p1 ~ 0.529)`... (in pts: +3). [U]
- Iterative deepening under time: our search is a single exact ply, so "deepening" = adding the contact extension (idea 9) and anytime candidate ordering
  (idea 20). pb4 found beam width > threshold and time > ~50% brought nothing [V] -- don't spend time budget by default; spend on contacts only.
- Transposition tables: our states rarely repeat (stochastic opponents), so skip TT; cache STATIC things (terrain distances) instead.
- Tablebases: idea 14. Opening books: idea 15 (weak expected value here).
- AlphaZero-style MCTS: not feasible in 45 ms pure Python; the neural value/policy are what could be distilled (idea 6).

### 3.3 Fog-of-war game AIs and bot contests
- AverageJoe / Superhuman generals.io (arXiv 2606.23348) [V]: persistent-visibility layers, enemy-sighting memory, 7-step army-delta history,
  2 temporal tokens of opponent land/army history over 512 steps; one attention head learns a belief over the enemy general, prior from map generation ->
  narrows -> converges; 25% top-advantage filtering; EMA +30 Elo; gamma=1 (no stalling when losing). Hand-crafted memory planes beat learned recurrence [V, Straka blog].
  Straka's blog [V]: emergent snowballing/backdooring/feinting; round 1 used 3-agent pool + BC on 16k human games (9M moves), shaped reward.
  Transfer: ideas 1, 7, 19; gamma=1 lesson = never reward stalling (our draw avoidance 16).
- AlphaStar: league with main agents, main exploiters, league exploiters [V snippet] -> idea 11, 25. Scouting/build-order timing: our `scout_move`,
  castle timing `castle_start`, strike windows [SR§2.4].
- Battlecode (MIT): postmortems from many years (2017-2025) report that the same general game plan was used by all top teams and gimmicks/micro
  (kiting, pathfinding, grouping, rush defence by spawning defenders) differentiated them [V snippet]; the PDFs themselves were not machine-readable
  here, so details are UNVERIFIED. Lessons are generic: scripted micro + tuning + rush defence (idea 17).
- Lux AI S2: top 3 were logic/planning, #1 (ry_andy_) used persistent unit roles and simulated chosen actions 5 to 50+ steps ahead within the call
  budget [S]; #4 was a deployment-aware RL system [S]. Our analogue: persistent "cycle" roles (we have `cyc`) and short simulation of OUR OWN plan
  for sanity (we do for the opening). Possibly extend: forward-simulate our own planned gather+launch ignoring the enemy to verify the arrival
  turn vs the 50-turn bonus (AS§1.7).
- Halite III: top bots were heuristic + local simulation; one top bot ran randomised navigation simulation and picked the best; zxqfl was among the top 4 [S].
  Message: cheap randomised local search over our own moves with an eval is competitive with deep logic -- supports idea 9/20.
- CodinGame: Agade (Ghost in the Cell winner, ~700 lines C++) used greedy per-entity heuristic scoring, a self-play look-ahead only for single
  decisions (production increase), and an Arena with CIs [V]; pb4 (Fall Challenge 2020) used DUCT (simultaneous-move UCT) for the draft and beam
  search for the main phase; time cap at 50% cost nothing [V]. Code Royale: Agade 3rd place postmortem exists [S, not read].
  Message: per-decision local simulations beat global search; Elo-CI-driven iteration is the true differentiator.
- Poker/Hanabi/OpenAI Five: belief states and deception matter only vs adaptive opponents [U]; not applicable beyond idea 18/0.1.

### 3.4 Offline learning tricks
- Distill the BC clone into rules: ideas 6 and 5 (value and option priors). Prefer logistic/small GBT (we already embed those) with hand features.
- Exploiter and PBT: ideas 11, 25. Parameter averaging: idea 3. Advantage filtering analogue: weight BC/threat training samples by outcome impact
  (decisive strikes, defended attacks) rather than uniformly (+ maybe small gain; test only with idea 5 data, same pipeline).
- Sparse vs shaped: idea 22.

### 3.5 Robustness
- Never forfeit: idea 8. Simple-bot floor: POOL_SIMPLE >= 0.95. Draws: idea 16. Anti-rush: 17. Deathtouch endgame: 14.
- Avoid single-opponent catastrophes: no-regret rule above; report min over opponents beside the mean in every A/B.

## 4. RECOMMENDED ORDER (my suggestion, with CPU cost)
1. Idea 8 gate + idea 10 SPRT (no bot risk, protects everything else) -- 1 h.
2. Idea 3 and 2 (post-processing/analysis of existing tuning logs) -- free CPU, minutes.
3. Idea 7 and 1 (parameter-gated, 5 and 8 specs) -- ~4-6 h of 3 workers each.
4. Idea 4 GATE matrix (2400 games) -- decides whether to build the classifier; same matrix gives the idea-12 meta-game data (reuse!).
5. Idea 5 and 6 (offline data work on replays, one shared feature pipeline) then A/B.
6. Idea 9/21 contact search; idea 11 exploiter run overnight.
7. Idea 14 only if Q9 shows >= 5-10% of games reach t=800.

## 5. SOURCES (URL; V = page content read, S = search snippet only)
- [V] Superhuman AI for Generals.io Using Self-Play RL, https://arxiv.org/html/2606.23348v1
- [V] Straka, Superhuman Generals.io agent blog, https://kam.mff.cuni.cz/~straka/blog/generals.html
- [V] Agade, Ghost in the Cell postmortem, https://github.com/Agade09/Agade-Ghost-in-the-Cell-Postmortem/blob/master/Agade_GitC_Postmortem.md
- [V] pb4, CodinGame Fall Challenge 2020 postmortem, https://github.com/pb4git/Fall-Challenge-2020
- [S] DeepNash, Mastering Stratego with Model-Free Multiagent RL, https://arxiv.org/abs/2206.15378
- [S] PSRO survey / Pipeline PSRO / empirical game-theoretic analysis, https://www.alphaxiv.org/abs/2403.02227 ; https://arxiv.org/pdf/2006.08555 ; https://arxiv.org/pdf/2403.04018
- [S] AlphaStar league (main/league exploiters), https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; https://www.alexirpan.com/2019/02/22/alphastar-part2.html
- [S] Lux AI S2 1st place (ryandy), https://github.com/ryandy/Lux-S2-public ; Kaggle writeups https://www.kaggle.com/competitions/lux-ai-season-2/writeups/bart-von-meijenfeldt-15th-place-a-goal-based-logic
- [S] Battlecode postmortems (PDF not machine-readable, UNVERIFIED details): https://battlecode.org/assets/files/postmortem-2019-oak.pdf , https://battlecode.org/assets/files/postmortem-2025-spaark.pdf , https://battlecode.org/assets/files/postmortem-2023-dont-at-me.pdf , https://medium.com/@thestuart/the-lessons-i-learned-from-mits-battlecode-2017-competition-post-mortem-570acdc6c1a2
- [S] Halite III postmortems, https://recursive.cc/blog/halite-iii-postmortem.html , https://mlomb.dev/blog/halite-iii-postmortem , https://github.com/TheDuck314/halite2018
- [S] Code Royale postmortems, https://github.com/Agade09/Agade-Code-Royale-Postmortem , https://github.com/robostac/cg-code-royale-postmortem , https://github.com/devYaoYH/cgPostMortem
- [S] Artificial Generals Intelligence, https://arxiv.org/html/2507.06825v2 (Human.exe described as ~30 prioritised plans over a belief state)
- [U] Texel tuning, SPSA, fishtest SPRT, CFR/regret matching: standard references (Chess Programming Wiki, Fishtest docs); not fetched.

# strategy-research (agent file)

Scope: web/literature sweep for ideas to strengthen our heuristic bot. Web research plus reading `bots/participant.py` and `sim/engine.py` only; nothing executed.
Legend: [V] verified from a fetched page, [S] seen only in a search-result summary, [I] my inference, [U] unverified.
Honest note up front: public, citable expert strategy for THIS ruleset (built castles, deathtouch 800) barely exists. The human guides assume the standard game (cities, no deathtouch). Most items below are transfers plus inference; the evidence quality column says which.

Engine facts used in the reasoning [V from sim/engine.py L248-253]: land tiles gain +1 army every 50 turns; general and castles gain +1 every 2 turns. So the human "25 rule" becomes a "50 rule" here (our bot's `open_end=50` and `open_div` already encode this). A castle yields 0.5 army/turn for >= 35 army, i.e. ~70 turns payback before the distance surcharge. The action space is ONE move per turn (`[kind,row,col,dir,split]`), so everything is a tempo game: a forcing threat costs the opponent its only move of the turn [I].

## 1. Ranked idea table

Impact = my guess at Elo/score gain for us against the likely field (RL/BC nets and mid-tier heuristics). Effort in our code, pure Python, <45 ms.

| # | Idea | Why it should help | Impact | Effort | Evidence / source |
|---|---|---|---|---|---|
| 1 | **Local exact tactical search (2-3 ply, top-K opponent replies) around contact zones and the general** | We have exact 1-turn resolution only (`resolve`, `safe`). A 2-ply "my move x their plausible replies" catches 2-step general kills, double-threats, and deathtouch chases that a 1-turn veto misses. Kore/Halite/Fall-challenge winners all lean on forward simulation of a local state | High | M | Kore top-3 were rule-based with forward simulation and a space-time threat ledger [S]; pb4 Fall 2020 beam search, >~30k sims gave no extra Elo [S, github pb4git] |
| 2 | **Deathtouch endgame solver (turn >= 760)**: exhaustive search over "who can touch whom" using the exact rule: any valid move (source army >= 2) onto the enemy start-general cell wins; only defence is capturing the attacker's source from a third tile | The game is decided by a small combinatorial race; exhaustive search on cells within ~6 of either general is cheap and exact. RL nets are likely under-trained here (their games mostly end < 600; our replay stats say ~7% pass 800) [I] | High (when games reach 800) | M | training-research.md Q8; sim/engine.py deathtouch step; Sentinel doctrine only says it "enables a decisive endgame" [V] |
| 3 | **Race calculator before defending** ("who kills first"): compute earliest arrival turn and size of our best strike on their general vs theirs on ours, including growth and the single-move constraint. If we win the race by margin, skip defence and counter-strike; if not, defend minimally | Over-defending is the classic heuristic bot leak; Sentinel's own loss audits show exactly "reinforce, release, repeat" and ignoring a large visible army outside its short horizon [V]. A race estimate fixes both directions | High | M | relh failures.md [V]; EklipZ "efficient attack timing" [V] |
| 4 | **Castle-build detection => strike window**: we already read exact enemy build prices. After a detected build, the enemy just paid >= 35 army; if the funds came from the home stack, home is weaker for ~N turns. Trigger a pre-staged stack toward the predicted general ring when our arrival beats their regrowth | Uses information RL bots must learn implicitly; mode fixation (castle-taking while ignoring attack) is documented for RL agents | Med-High | S-M | arXiv 2507.06825 mode-fixation [V via training-research]; funding-source assumption [U] (check engine `apply_build_actions`) |
| 5 | **Fog army likelihood map (EklipZ-style)**: maintain a decaying heat map of enemy stack locations; when an enemy tile appears from fog, BFS back into fog scoring fog tiles by distance to the exit point; feed the map to `learned_threat` and the general belief | EklipZ calls this "insanely accurate" even vs concealment [V]; we have `tracks` (track_ttl 30) but not a diffusion belief. Improves defence timing and general location together | Med-High | M | github EklipZgit/generals-bot README [V] |
| 6 | **Opponent-type classifier and parameter profiles**: from turns 20-150 measure land curve, spawn-direction of expansion, first castle turn, first stack contact; select among 3-4 CMA-tuned parameter sets (rush-counter, turtle-punish, default) | Lux S3 meta-learning lesson: the winners adapt to the opponent within a match [S]. Tuning pool already has anchors per style, so profiles are tunable offline | Med-High | M-L | Lux S3 summary [S]; training-research rec 1 anchors |
| 7 | **Attack timing aligned to ticks**: land +1 at t%50==0, general/castles +1 at even turns. Time the final arrival/capture so our stack lands the turn after a tick and capture-race parity (chasing > reinforcing > smaller army) works for us | Cheap exact edge in contested tiles; EklipZ optimises "maximum damage each army bonus timing" [V] | Med | S | EklipZ README [V]; engine ticks [V] |
| 8 | **Forcing-move tempo play (feints)**: because the opponent also has one move per turn, an attack that forces a defensive response costs them tempo; use short feints (stack approaches then retreats) against deterministic nets that respond identically to identical observations | Documented exploitability of greedy RL policies [V quant-eagle README]; AverageJoe learned feints against humans [V Straka blog] | Med | M | Straka blog [V]; quant-eagle [V] |
| 9 | **Cutting / choke control**: capture or hold single-tile corridors on the enemy's gather paths and on ours (graph articulation points between predicted enemy position and our general) | EklipZ "blocks choke points using basic graph theory" [V]; human guides stress cutting supply lines [S] | Med | M | EklipZ README [V] |
| 10 | **General concealment discipline**: keep traffic near the general minimal, route gathers to cross the general only when necessary, expand "hollow" around it, never attack from the general tile | Sentinel doctrine "obscure the location of our capital, avoid a trail of movements" [V]; human tips [S]; baseline conservative strategy in generals.bot docs (never attack from the general tile) [S]. We have `gen_move_pen=0.6` already, so tuning rather than new code | Med | S | relh playing-doctrine [V] |
| 11 | **Enemy-land priority**: taking enemy land swings land difference by 2 per capture vs 1 for neutral | Wiki 1v1 guide [V]. Our `v_enemy=2.2` already ~matches; just confirm CMA keeps it >= 2 | Low (done) | S | wiki.generals.io 1v1guide [V] |
| 12 | **Second-round flank opening principles**: expand toward the least-explored flanks, claim land between general and enemy (enemy must cross our land), watch opponent land counter to infer opening | Wiki [V]; mostly encoded in opening sim and belief | Low-Med | S | wiki [V] |
| 13 | **Land-counter inference of the enemy opening**: enemy land at turn 50 tells you whether it executed a ~25-land opening; combine with spawn belief to narrow general location | Wiki [V]; our belief uses land counts already | Low | S | wiki [V] |
| 14 | **Beam search over gather/castle plans** (width ~50, depth ~15 plan-steps) using a cheap sim instead of greedy trees | pb4 beam search dominated Fall 2020 [S]; but our gather already uses pruned MST-like trees (EklipZ idea). Gain is modest, cost in CPU is real | Low-Med | L | pb4git [S] |
| 15 | **Determinised MCTS** for contested mid-game only | Literature-standard for fog games, but in pure Python ~45 ms gives maybe 200-500 playouts, too few for a 5-action-per-army branching; idea 1 dominates it | Low | L | general knowledge [I] |
| 16 | **Anti-RL: late-game probing of mountain dead-ends** | RL agents get stuck in mountain pockets [V arXiv 2507.06825]; attack so their response path must detour | Low-Med | M | training-research Q8 [V] |
| 17 | **Castle defence / castle recapture priority**: top bots hold castles consistently, weak bots surrender them; defend ours with local garrison, snipe undefended enemy castles (`v_ecastle` already 8) | Mattz post: weakness = poor castle defence and early pressure response [V] | Med | S | Mattz Substack [V] |
| 18 | **Early scout/pressure**: small scouts at start that find the enemy king early (human tips); we have `scout_start=60` | Human tip [S]; belief value of early contact is high because it collapses the spawn prior | Low-Med | S | notebook.neelr.dev [S] |
| 19 | **Anytime safety net**: wrap decide() in a deadline check with a cheap fallback (best legal safe move) and keep forfeits at zero | Zero-forfeit is a real edge vs NN entrants that may time out on one shared core [I]; our `soft_budget_ms=45` exists. Verify the fallback path is tested under load | Med | S | training-research Q8 point 6 [I] |
| 20 | **Regression-chain discipline** (not a bot feature): keep every shipped version as a league opponent, accept changes only via SPRT | Kore winner "long regression chains" [S]; Kaggle RL winners mostly rule+sim [S via training-research] | Process | S | Kore writeups [S] |

Dropped as low value: city-timing from human guides (no neutral cities here, our castle economy already parametrised and tuned by CMA); multi-army splitting "three shards" (one move per turn makes it expensive, and Straka's agent reportedly used it because it has precise belief [V blog]).

## 2. Details and pseudocode for the top 6

### 2.1 Local tactical search (ideas 1 and 7)
Goal: replace "veto if the 1-turn resolution loses the general" with "choose among candidate moves by minimax over a small local window".

```
def tactical_pick(B, my_candidates, deadline):
    # window = cells within R=6 of either general or of any visible enemy stack >= 5
    S0 = B.snapshot_local()                      # arrays for window only; outside window frozen
    opp_set = enemy_reply_set(S0)                # <= 6 moves: each visible enemy stack's
                                                 # forward/toward-our-general/capture moves
                                                 # + 'pass' + belief-weighted fog entry
    best = None
    for a in my_candidates[:8]:                  # heuristic top 8 from decide()
        worst = +INF
        for b in opp_set:
            S1 = resolve(S0, a, b)               # existing exact resolve (chasing>reinforcing>smaller)
            v = eval_local(S1)                   # then my greedy reply for 1 more ply:
            if not terminal(S1): v = max(eval_local(resolve(S1, a2, pass_like(b2)))
                                         for a2 in top3(my_moves(S1)) ...)
            worst = min(worst, v)
            if time() > deadline: break
        best = max(best, (worst + prior_bonus(a), a))
    return best.a
# eval_local = +big if enemy general captured, -big if ours, else
#   (my_army_near_gen - enemy_army_reach_gen_in_k) + w*land_delta + tick-parity bonus
```
Budget: 8 x 6 x ~2 ply = ~100 resolve calls on a 13x13 window; our `resolve` is O(1) per move pair, fine (<5 ms). Only fire when an enemy unit is within `threat_vis_range` or t >= 740, so the cost is amortised. Evaluate in arena as a toggle param `tactical_on`.

### 2.2 Deathtouch solver (idea 2)
Rule recap: from t >= 800 a valid move whose destination is the opponent's start-general cell wins instantly; both touching on the same turn is a draw [V engine]. A valid move needs source owned with army >= 2.
```
def dt_solve(B):
    # state: cells within 7 of either general; ours/theirs armies; 1 move per side per turn
    # Attack check: for each of my cells c adjacent to enemy G with army >= 2 -> win now.
    # Threat check: for each enemy cell adjacent to my G with army>=2 -> I lose unless
    #   (a) I move onto that source cell with army > its army (capture it) -- 'third-tile defence'
    #   (b) I win this turn (draw/win ordering: check resolve)
    # Search: depth-4 alpha-beta over (my move, their move) pairs restricted to:
    #   moves into/along the ring (radius 2) of each general, and stack-feeding moves
    # Terminal eval: win/loss/draw; non-terminal: 
    #   dist-to-contact race: min over my cells (arrival_turn(c->G_enemy) with army>=2)
    #   minus the same for them; fortress bonus = number of my cells adjacent to G with army >= enemy_max_reach
```
Practical heuristics that fall out: (1) before 800 build a ring: keep each general-neighbour cell at >= (enemy local max reachable stack)+1; (2) at 800+ never leave a general-adjacent cell empty of a defender that can recapture; (3) approach the enemy general along a path that places TWO >=2-army cells adjacent so one capture cannot stop both. These are untested [I]; the solver validates them in sim before shipping. Also prefer the exact solver result to `fortress_turn`/`dt_stage_turn` constants if time permits.

### 2.3 Race calculator (idea 3)
```
def race(B):
    # our best strike: for each of our stacks s: arrive_t(s -> enemyG_candidates by belief)
    #   army_at_arrival = s.army + growth along path (only tiles we pass over don't add; count the
    #   general/castle ticks only if the stack sits on them) - 1 per step lost to path tiles = 0 (moves leave 1 behind)
    # their best strike symmetric using known/visible stacks plus worst-case hidden stack
    #   = max(hidden_stack_bound, track estimates)
    my_T,  my_A  = best_strike_us()
    en_T,  en_A  = best_strike_them()
    # kill requires A_at_arrival > defender_army_at_arrival (+1)
    if my_T + 1 < en_T and my_A > def_est(enemyG, my_T): return 'GO'   # ignore low-threat defence
    if en_T <= my_T + 2 and en_A > my_garrison_at(en_T): return 'DEFEND_MIN(need=en_A-garr+margin)'
    return 'NEUTRAL'
```
Use: modifies `garrison_need()` (reduce when GO, set exact when DEFEND_MIN) and bumps `w_launch`. Because the single-move constraint means gathering reinforcements costs moves, include "moves needed" (= path length of the reinforcement) in my_T/en_T for the defender side.

### 2.4 Castle-build strike window (idea 4)
Check `apply_build_actions` first: who pays for the castle and from where [U: confirm the army is deducted from the builder tile]. Then:
```
on detected enemy build at cell X with price P at turn t0:
    mark enemy 'weak_until' = t0 + P/ (enemy_income) ... or simply t0 + 25
    if our stack S exists with arrive_t(S -> ring around X) <= weak_until and S.army > est_garrison(X)+margin:
        set cycle target = X (castle) ; v_ecastle already 8 -> raise to 12 inside window
```
Cheap and testable: league against ResBot-BC / hunter_castles (which build castles) and count castle snipes.

### 2.5 Fog army likelihood map (idea 5)
```
H[cell] = P(enemy 'main stack' at cell), init at belief of enemy general
each turn: if enemy stack seen at c: H = delta(c) mix (we know position)
           elif an enemy army exits fog onto visible cell c with army a:
                for fog cells f: H[f] *= exp(-(bfs(f,c) - k)^2/2) with k ~ steps since last seen
           else: H = (1-eps)*diffuse(H) + eps*prior;  H[visible cells] = 0
threat(cell) feeds learned_threat feature 'mass of H within r of our general / stack';
general belief += alpha * H-weighted source pointer
```
Pure Python diffusion on a 21x21 grid: ~441 cells x 4 neighbours = under 1 ms. Keep tracks as the discrete core; H only fills fog gaps.

### 2.6 Opponent-type profiles (idea 6)
```
features at t=120: my_land - opp_land_seen, enemy first-contact turn & dir, #castles built by enemy so far,
                   enemy land at 50 (opening quality), enemy stack growth at general (hidden), moves/turn variance
class = argmax over {rusher, turtle/castler, expander, net-like} using a small decision rule fit offline
        on arena logs vs the anchor pool (logistic regression on 6 features)
profile[class] = CMA-tuned deltas on ~6 params: garrison_frac_hidden, castle_every, launch_max, w_launch, attack_min_army, scout_max_frac
```
Offline: tune 4 profiles by running CMA only against that class pool (anchors already present). Gain is real only if profile-specific optima differ materially; check by comparing optimum parameters from the existing per-opponent scores before building the classifier [I].

## 3. Findings per research area (short)

Human expert strategy [V wiki 1v1 guide]: 25-land opening (here 50-turn periods, land tick 50 turns); memorised 13/11/10/14-starts; explore several directions at once, claim land between your general and the enemy so attacks must cross your land; flank-expansion; track enemy land counter; maximise "free army" = army - land; enemy captures swing 2 vs neutral 1. Human tips [S, gist/notebook]: deception (little traffic near king), diversion/pincer from the side, scouts early, hail-mary back-door by following low army numbers from a weak point to a general/castle, hiding the general via surface area. Nothing found on reddit/youtube at fetchable depth (searches returned no usable pages; [U] there may be better material behind logins).

Bot community [V]: EklipZ Human.exe is pure algorithmic: MST gather with pruning, army tracker, fog-exit BFS general prediction, choke-point interception, DFS/BFS hunts recomputed each turn, map-middle preference [github README]. Straka's neural agent beat Human.exe 100% and shows feints, backdoors, three-way splits, encirclement; Human.exe had "roughly thirty precomputed plans" [Straka blog, V]. So a fully heuristic bot has a ceiling vs a strong net, but nets must run in 150 ms on one core here.

generals.bot competition [V]: Marathon ranking in bot-scout.md. Mattz: rules-based approach gave poor initial results, BC alone insufficient, PPO plus targeted castle examples worked; his weakness was castle defence and early-pressure response [Substack]. Organisers state expansion tempo and castle timing matter [about page]. No write-ups of ResBot/nanomena/Kubic/FreeLunch found (bot-scout confirms).

Other competitions: Kore 2022 top-3 were rule-based with forward simulation, threat ledgers and staged defence/attack/economy phases [S]; Halite IV 4th place rule-based with heuristic placement [S]; Lux S3 rewards in-match opponent adaptation [S]; CodinGame Fall 2020 winner beam search, speed beyond ~30k sims irrelevant [S]. Common separator: exact simulation of threats + disciplined staged priorities + heavy local regression testing, not fancy search.

Exploiting opponent types (inference [I] unless cited): 
- Expander/Hunter baselines: they ignore defence timing; a race-calculator GO (idea 3) plus early stack strike beats them; keep this since ours already wins >90% presumably.
- BC clones (ResBot-style): deterministic response to observations at temp 0, thus feints (idea 8) and castle-window strikes (idea 4).
- RL nets: mode fixation, mountain dead ends, weak late game [V arXiv 2507.06825]; use ideas 2, 4, 16.
- All-ins/rushes: garrison from hidden-stack bound (we have), race calculator to counter-strike when their stack left home, and 2-ply tactics around the general.

## 4. Suggested order (value/effort)
1. Idea 3 race calculator (M, high) and idea 4 castle window (S-M) - both sit on top of existing belief/track data.
2. Idea 1 tactical 2-ply search gated by proximity, reusing `resolve`.
3. Idea 2 deathtouch solver if replay stats show games reaching 800 against the likely field; otherwise lower.
4. Idea 5 fog heat map (feeds both threat model and belief).
5. Idea 6 profiles only after verifying parameter optima differ per opponent class.
Tune new gates through the existing CMA params (`tactical_on`, `race_margin`, `castle_window_len`).

## 5. Sources
- Generals.io wiki 1v1 guide: https://wiki.generals.io/1v1guide.html [V]
- twolfson strategies gist (404 on fetch; seen in search only): https://gist.github.com/twolfson/4bbbb40bd8b7ed670694b5a4dae04931 [S]
- on generals.io tactics: https://notebook.neelr.dev/stories/on-generals-io-tactics [V, thin]
- EklipZ bot README: https://github.com/EklipZgit/generals-bot [V]
- Straka, Superhuman Generals.io agent: https://kam.mff.cuni.cz/~straka/blog/generals.html [V]; paper https://arxiv.org/html/2507.06825v2
- Mattz, Trying to Win an AI Bot Competition: https://eventwaves.substack.com/p/trying-to-win-an-ai-bot-competition [V]
- generals.bot about page: https://www.generals.bot/about [V]
- relh Sentinel doctrine and failure audits: https://github.com/relh/generals-bots/blob/main/docs/agent-development/playing-doctrine.md [V]; local copy vendor/ext/relh_generals-bots/docs/agent-development/failures.md [V]
- quant-eagle RL bot: https://github.com/quant-eagle/generals-competition-rl-bot [V]
- Kaggle RL competitions writeup (page body did not load): https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio [S]
- pb4 Fall Challenge 2020 postmortem: https://github.com/pb4git/Fall-Challenge-2020 [S]
- Lux S3 solutions (3Comets etc.): https://www.kaggle.com/competitions/lux-ai-season-3/writeups/3comets-multi-agent-rl-silver-solution-by-3comets- [S]
- Halite IV 4th place: https://github.com/0Zeta/HaliteIV-Bot [S]
- Local: sim/engine.py (tick rules, deathtouch), bots/participant.py (PARAMS), agents_shared/training-research.md, agents_shared/bot-scout.md

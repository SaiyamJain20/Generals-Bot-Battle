# algo-study — algorithms to port into bots/participant.py

Status: DONE (reading task; no games run). Ranked list: section 4 at the bottom.
Owner: algo-study agent. Sources read are listed per section. External code is reference only
(nothing copied into the submission; ideas re-implemented).

## 0. Context recap (what our bot already has, so ideas are framed as deltas)

Ruleset facts that drive everything below (sim/engine.py, generals/core/env.py "competition"):
- Board 18..21 x 18..21 (independent h, w), 18-26% mountains, spawn BFS distance >= 17.
- Neutral plain = 0 army, so 1 army captures a neutral tile. No neutral cities at all.
- Land bonus +1 on EVERY owned tile every 50 turns (t % 50 == 0); general/castles +1 every 2 turns.
- Castle cost 35 + sum over own structures (general + castles) of max(0, 14 - 2d), d = manhattan.
  => a castle at d >= 7 from all own structures costs 35; adjacent to the general costs 35+12 = 47.
  A castle pays 1 army / 2 turns -> 35 army payback after 70 turns (plus land bonus is unaffected).
- Deathtouch from turn 800: ANY valid move onto the opponent's starting general cell wins.
  Truncation at 1200 = draw.
- Move order: chasing > reinforcing > smaller source army > lower player index.

Our Bot (bots/participant.py) hooks that the ideas below plug into:
- `update()` exact army accounting (enemy builds and their price -> rings on the enemy general);
  `update_tracks()` = greedy nearest-match of visible big enemy cells, TTL 30, no fog extrapolation.
- `garrison_need()` -> need_g + threat_eta from visible cells within 10, tracks within 12, and a
  "hidden army" estimate scaled by fog distance. `gather_move(g, need, budget=eta-1)` serves it.
- `gather_select()` greedy root-ward path additions by army/move ratio (an EklipZ-style greedy
  approximation, not knapsack, no pruning pass).
- `best_capture()` single-step capture scoring; `home_fill_move()`; `scout_move()`.
- `cycle_move()/new_cycle()` = gather into the general (or castle site) then launch half/all to
  `choose_target()` via Dijkstra `path_to()`.
- `castle_build_now()/castle_site()` with fixed cadence `n_castles_wanted()` (1 + (t-90)/45).
- `try_kill()` path-cost kill check vs estimated general army; `endgame()` fortress + deathtouch.
- `resolve()/safe()` 1-ply exact check vs adjacent enemy moves into the general only.

Timing note: this ruleset's clock is the same as generals.io's in ticks. Each player makes 1 move per
tick, the general/castles get +1 every 2 ticks (generals.io: cities +1 per "turn" = 2 ticks), and all land
gets +1 every 50 ticks (generals.io: every 25 "turns" = 50 ticks). So EklipZ's per-tick, 50-tick-cycle
logic (`map.turn`, `remainingCycleTurns`, "cycle = 50") maps 1:1 onto our `turn`, and its city
heuristics map onto built castles. The only differences: castles are built for 35+ and never neutral,
and there is deathtouch/draw.

Economic arithmetic that should drive the priorities:
- 1 tile = +1 army per 50 turns for the rest of the game. At t=200 that is ~20 army by t=1200, for a
  cost of 1 army and 1 move. With 150 tiles that is +3 army/turn, which is about 6 castles. **Land is
  the biggest economic lever**, and enemy tiles count double (we gain, they lose).
- 1 castle = +0.5/turn for 35..47 army plus about 10-20 gather moves. It pays back in 70-94 turns.
  Per move it is about as good as land (0.5/15 ≈ 0.033/turn per move vs 0.02 per tile-move), and
  per army it is worse (0.0125 vs 0.02). Castles compound, so earlier is better while it is safe.

---------------------------------------------------------------------------------------------------

## 1. EklipZ (Human.exe), vendor/ext/EklipZgit_generals-bot: extracted algorithms

(Licence: see License.txt in the repo. Ideas only; nothing copied.)

### 1.1 Opponent move classifier + fog "gather queue" → concentrated-stack estimate  [VALUE: HIGH, effort M]

**(a) What it does and why it wins.** EklipZ never assumes that *all* hidden army can hit it. It tracks
how many turns the opponent has spent gathering in the fog versus expanding in the fog, and turns that
into the size of the stack the opponent has *actually* been able to concentrate. The enemy gets one move
per tick, so a stack can only grow by about one tile's value per non-capture move. The result is
`approximate_fog_army_available_total` ("fog risk"), and it drives defense, the "greedy turns
available" budget, and whether to launch.

Our `garrison_need()` instead uses the hidden army bound
(`opp_army - visible - (opp_land - vis_cells - 1)`) times a fraction picked by fog distance. That
causes both of our failures:
- **Too paranoid** when the enemy has spent its moves expanding. We keep up to 50% of our army at
  home, so we out-expand less.
- **Too lax** when the enemy has quietly gathered a single stack (the fraction cuts it to 15-50%).

**(b) Refs.**
- `Strategy/OpponentTracker.py`:
  - `FogGatherQueue` L21-235 (a histogram of fog tile sizes; `increment_army_bonus` L84,
    `pop_next_highest` L100).
  - `_check_missing_move` L956 (classifies the unexplained move).
  - `_assume_fog_gather_move` L1061 (stack += popped tile - 1).
  - `_assume_fog_empty_tile_capture_move` L1089 (spends a 2-tile, else 1 from the stack).
  - `_execute_emergence` L1619 (a visible emergence > 0.87·estimate ⇒ reset the fog estimate).
  - `_handle_vision_losses` L1348 (a big tile that goes into fog counts as immediately available).
  - `get_approximate_fog_army_risk` L1835 (risk in N turns = current estimate + future pops from the
    queue + castle income).
- `BotModules/BotDefense.py`:
  - `check_fog_risk` L1838 (high_fog_risk if risk > army on the defensive spanning tree).
  - `determine_fog_defense_amount_available_for_tiles` L43 (subtracts tracked fog armies that cannot
    reach the target within N turns).

**(c) Pseudocode for our Bot** (new `EnemyEcon` state, updated at the end of `update()`; it replaces the
`hidden` block of `garrison_need()`):
```python
# state (init in __init__)
self.fq = [0]*64          # fq[v] = number of enemy FOG tiles believed to hold v army (v>=1)
self.fog_stack = 0        # S: army the enemy has concentrated in fog (excl. general)
self.en_move_hist = []    # last 50 classifications, for stats / aggression

def classify_enemy_move(self, prevT, prevO, prevA):
    """Exactly one enemy action per tick. Returns 'vis', 'fogcap', 'foggather', 'build'."""
    if self.built_this_turn:                 # set by the existing accounting when built >= 35
        return 'build'
    # visible move: a cell visible on both ticks that was enemy-owned and whose army dropped
    # (growth-adjusted) with no move of ours onto it, or a cell newly flipped to enemy
    # (neutral->enemy, or our cell lost to a move we didn't cause).
    for i in self.vis_both:                  # cells visible at t-1 and t (cheap: keep a list)
        if prevO[i] == 2 and self.O[i] == 2 and self.A[i] < prevA[i] + self.grow(i):
            if i != self.my_last_dst: return 'vis'
        if prevO[i] != 2 and self.O[i] == 2: return 'vis'
    dland = self.opp_land - self.prev_opp_land + self.enemy_tiles_we_took_this_turn
    return 'fogcap' if dland >= 1 else 'foggather'

def update_fog_queue(self, kind):
    fq = self.fq
    if self.turn % 50 == 0:                  # land bonus: every fog tile +1 (shift histogram)
        fq[1:] = fq[:-1]; fq[0] = 0
    if kind == 'foggather':                  # pop the biggest fog tile into the stack
        v = max((k for k in range(len(fq)) if fq[k]), default=0)
        if v >= 2:
            fq[v] -= 1; fq[1] += 1; self.fog_stack += v - 1
    elif kind == 'fogcap':                   # a 2 captures a neutral -> two 1s; else the stack pays 1
        if fq[2]: fq[2] -= 1; fq[1] += 2
        else:     fq[1] += 1; self.fog_stack = max(0, self.fog_stack - 1)
    # keep the histogram size consistent with the number of fog tiles
    # fog plain tiles only: general and castles are modelled separately (they grow 1/2 ticks)
    gen_vis = self.egen >= 0 and self.T[self.egen] == 4
    fog_castles = sum(1 for c in self.enemy_castles if self.T[c] in (0, 5))
    nfog = self.opp_land - self.vis_enemy_cells - (0 if gen_vis else 1) - fog_castles
    cur = sum(fq)
    while cur < nfog: fq[1] += 1; cur += 1           # unseen captures, vision losses of 1s
    k = 1
    while cur > nfog and k < len(fq):                # reveals: drop the smallest
        take = min(fq[k], cur - nfog); fq[k] -= take; cur -= take; k += 1
    # cap by the hard bound: the enemy cannot have concentrated more than the hidden standing army
    hidden = self.opp_army - self.vis_enemy_army - max(0, self.opp_land - self.vis_enemy_cells)
    self.fog_stack = min(self.fog_stack, max(0, hidden))

def on_enemy_stack_seen(self, v):            # called from update_tracks() for a NEW big track
    if v >= 0.87 * (self.fog_stack + 1):
        self.fog_stack = 0                   # the concentrated army has emerged ("full fog reset")
    else:
        self.fog_stack = max(0, self.fog_stack - v)
# When a tracked stack walks into fog, do NOT add it to fog_stack: it stays a track (1.2).

def fog_risk(self, in_turns=0):
    """Army that could hit us in `in_turns` turns: stack + general + castles + future pops."""
    gen = self.egen_army_est()               # last seen + (t - seen)//2, or accounting-based
    r = self.fog_stack + max(0, gen - 1) + sum(self.ecastle_army_est(c) for c in self.enemy_castles)
    tiles = sorted((v for v in range(len(self.fq)) for _ in range(self.fq[v])), reverse=True)
    for j in range(min(in_turns, len(tiles))):
        r += tiles[j] - 1
    return r
```
Use in `garrison_need()`: replace `hidden*f` with
`need = fog_risk(in_turns=max(0, eta - D_fog)) + margin` (D_fog = fog distance from the general),
but only count enemy castles and the general when they are within reach (BFS distance ≤ eta + 2).
The general and castles only threaten us if the enemy launches from them, so cap by `dist_g`.
Also expose `en_gather_ratio = #foggather / (#foggather + #fogcap)` over the last 50 ticks. If it is
> 0.6, the enemy is building a hit, so raise defense or hold the launch. If it is < 0.3, they are
expanding, so we can go greedy (expand or launch). That is EklipZ `check_gather_move_differential`
L1735.

Risks: the classification is wrong on ticks when our own move captured an enemy cell (we already know
that from `last_action`), and when the enemy moves a visible stack *into* fog (that counts as 'vis').
Our existing exact accounting (`R`, `built`) already supplies `built_this_turn`.
Cost: O(n) per tick.

### 1.2 Fog-aware army tracks: advance only on unexplained enemy moves  [VALUE: HIGH, effort S]

**(a)** When a tracked stack goes into fog, EklipZ moves it along its expected path (the shortest path
to our general, cities or targets), but only on ticks when that player's move was not seen elsewhere
(`player_moves_this_turn`). If the next tile on the path becomes visible and the stack isn't there, it
drops that path, and alternative routes are kept as "entangled" splits.

Ours (`update_tracks`) freezes the last-seen cell. `garrison_need` assumes worst-case advance
(`dist_g - (t - ts)`), which over-reacts when the enemy spent those moves elsewhere. It also never
reasons about *where* the stack is, so we cannot intercept it.

**(b)** `ArmyTracker.py`: `move_fogged_army_paths` L467, `_move_fogged_army_along_path` L3960,
`get_army_expected_path` L2425 (non-flank L2564 / flank L2651), `increment_fogged_armies` L446
(+1 on bonus), `find_fog_source` L1457 (when an army emerges, trace back the fog path that produced it).

**(c)** Extend a track to `[cell, army, ts, adv, path]`:
```python
def update_tracks(self):            # after classify_enemy_move()
    ...match visible big cells as now...
    moved_elsewhere = (self.last_enemy_move_kind in ('vis', 'build'))
    for tr in unmatched_fog_tracks:
        if not moved_elsewhere:
            tr.adv += 1                       # it MAY have moved one step
            nxt = tr.path[min(tr.adv, len(tr.path)-1)]
            if self.T[nxt] not in (0, 5) and not (self.O[nxt] == 2 and self.A[nxt] >= tr.army//2):
                # predicted cell is visible and empty -> it went another way: re-path from its
                # last plausible cell to the next target, avoiding visible cells
                tr.path = self.fog_path(tr.path[tr.adv-1], targets=[self.general] + list(self.my_castles))
                tr.adv = 1
        if self.turn % 50 == 0: tr.army += 1
    # threat distance for garrison_need / defense:
    #   d = max(1, dist_g[tr.path[tr.adv]])  instead of  dist_g[c] - (t - ts)
```
`fog_path` = BFS from the track cell through cells not currently visible (plus the target), toward our
general. If no fog-only path exists, fall back to the plain BFS path.

Also: when a new big enemy stack *appears* with no track, `find_fog_source` says it came from the
adjacent fog. Use that cell for the emergence general-belief update (1.6).
Cost: one BFS per fog track per re-path (≤ 8 tracks): fine.

### 1.3 Interception: threat-path multi-root defense gather with deadlines  [VALUE: HIGH, effort M]

**(a)** EklipZ does not gather defense *to the general*. A detected threat has a path
P = [p0 = enemy stack, p1, …, pL = general]. **Every tile on P becomes a gather root, with a deadline**
equal to the number of ticks before the enemy stack reaches it (`ThreatObj.convert_to_dist_dict`).
The defensive gather then collects army onto whichever path tile it can reach in time, so the enemy is
met as far up the path as possible. The gather excludes the path tiles themselves from the counted value,
and lightly prefers roots far from the general (`prioMatrix = 0.0001·dist_from_threat`).

On top of this, the ArmyInterceptor:
- enumerates intercept points among chokes common to all plausible threat routes (1.4);
- simulates "our army walks to x, the enemy walks toward the target" for each one (army after the
  collision, plus the econ value of the tiles saved);
- picks the best value per turn, with an optional required delay (wait for the enemy to step next to
  us so that we capture the stack *by chasing*, which takes move-order priority).

Why it wins vs "big stack through fog while spread out":
- gathering to the path catches the stack 2-6 tiles out, using army that is *between* the enemy and the
  general;
- a lone general garrison is exactly what loses when 50% of our army is in the field.

**(b)** Refs:
- `DangerAnalyzer.ThreatObj.convert_to_dist_dict` L36.
- `BotModules/BotDefense.get_gather_to_threat_paths` L640.
- `try_threat_gather` L716 (negatives = threat path tiles; prio = a tiny bonus for distance from the
  threat's start).
- `Behavior/ArmyInterceptor.get_interception_plan` L423, `_get_intercept_plan_options` L1150,
  `_get_value_of_threat_blocked` L1883, `_should_delay_or_split` L2538.
- Juraj's equivalent (see §2): "move to the choke if you get there first, else reinforce the general".

**(c)** Replace the garrison option in `macro()` when a binding threat exists (visible stack, or a track
from 1.2):
```python
def threat_path(self, src):
    # enemy's cheapest path to our general: Dijkstra with enemy-perspective costs
    # (our cells cost their army, neutral 1, enemy cells 1). Reuse path_to logic, flipped owner.
    ...
def defend_on_path(self, src, v):
    P = self.threat_path(src)             # P[0]=src ... P[L]=g
    L = len(P) - 1
    best = None
    roots = [k for k in range(1, L + 1) if self.O[P[k]] == 1]       # our cells on the route
    # restrict to chokes + general to bound the cost (1.4), at most ~5 roots
    roots = [k for k in roots if k == L or self.is_choke(P[k])][-5:]
    for k in roots:
        r = P[k]
        # enemy army when it arrives at P[k]: v + (enemy cells it sweeps) - (our cells it must beat)
        v_k = v + sum(self.A[P[j]] - 1 for j in range(1, k) if self.O[P[j]] == 2) \
                - sum(self.A[P[j]] for j in range(1, k) if self.O[P[j]] == 1)
        need = v_k + 1 - self.A[r] + (k // 2 if r == self.general else 0) * 0   # growth: gen grows +k//2
        if need <= 0:
            best = best or (k, None); continue
        path_cells = set(P[1:k])
        a, tot = self.gather_move(r, need=need, budget=k - 1,          # deadline = k-1 moves
                                  allowed=lambda j: j not in path_cells)
        if a and tot >= need:
            if best is None or k < best[0]:   # prefer the furthest-forward feasible intercept
                best = (k, a)
    return best[1] if best else None
```
Plus a direct **chase-kill** check before it (sentinel/juraj idea, §2.1). If one of our cells x is
adjacent to the stack cell e with A[x] - 1 > A[e], attack e now (it is never the general unless it's
the only option). Move order: when the enemy moves out of e, our move into e counts as "chasing" and
resolves first, so we fight the full stack *before* it moves. Use `resolve()` to confirm.

Cost: ≤5 `gather_select` runs with small budgets, plus one Dijkstra. That is about 5-10 ms in pure
Python on 400 cells; gate it on "binding threat exists".

### 1.4 Chokepoints (ArmyAnalyzer pathways) for defense placement and scouting  [VALUE: MED-HIGH, effort S]

**(a)** For two endpoints A (threat, or the predicted enemy general / spawn candidates) and B (our
general), with BFS maps dA and dB and D = dA[B]:
- the **pathway** is S = {x : dA[x] + dB[x] == D} (every shortest route);
- group S into layers by dB;
- a layer of size 1 is a "zero choke": every shortest route must pass it;
- a layer of size 2 whose cells share a common neighbour gives a "one choke", the shared neighbour,
  from which a defender reaches either cell in 1 move.

`chokeWidths[x]` = the size of x's (dA, dB) layer. `interceptTurns[x] = D - dA[x] + 1` is how many ticks
an interceptor has to be on x.

BoardAnalyzer's `innerChokes`/`outerChokes` is a cheaper single-source version on the dist-from-general
map: a tile with exactly one neighbour one step closer to the general is an outer choke. Its
`_get_defensive_choke_point` walks layers from the general toward the enemy and keeps the *furthest
layer that is fully visible and contiguous*. That is the defensive line where the garrison stack
should sit: it sees the attacker coming and covers all routes.

Why it wins:
- a stack parked on a choke 3-6 tiles out covers every shortest route with one stack;
- it gets more warning than the general does;
- a choke can always be defended one tick later than an open approach (from the EklipZ README).

hv (the RL fork) measured that 51% of enemy presence within 6 of our general sat on 3 cells.

**(b)** `ArmyAnalyzer.build_chokes_and_pathways` L~150, `build_intercept_chokes` L~205 (zeroChokes and
oneChokes), `BoardAnalyzer.rescan_chokes` L117, `_get_defensive_choke_point` L211.

**(c)** Run in `update()` when `belief_target()` changes, or every 10 ticks:
```python
def choke_analysis(self, A_cells):
    dA = self.bfs(A_cells); dB = self.dist_g
    D = min(dA[self.general], INF)
    layers = {}
    for x in range(self.n):
        if self.pas[x] and dA[x] + dB[x] <= D + 1:          # allow 1 slack (routes 1 longer)
            layers.setdefault(dB[x], []).append(x)
    chokes = []
    for k in range(2, min(D, 9)):
        L = layers.get(k, [])
        if 1 <= len(L) <= 2:
            chokes.append((len(L), -k, L))                # narrow and far is better
    self.chokes = chokes
    self.def_choke = chokes and min(chokes)[2]             # tiles of the best defensive layer
```
Uses:
1. In `garrison_need()`, defense army within 1 of `def_choke` counts as garrison when the threat's
   ETA to the choke is > our distance to it (hv's "still ahead" rule, §3).
2. In `home_fill_move()`, prioritise owning and seeing the choke layer (vision = warning).
3. As the gather root for the "defensive stack" between cycles: a castle placed on the choke is ideal
   (juraj places castles on approach lanes, §2).
4. In 1.3, use the choke layers as the candidate roots.

With multiple spawn candidates, use `A_cells` = the top-3 belief candidates.

### 1.5 Kill-threat search: "sweep-along-path" DP (both directions)  [VALUE: MED-HIGH, effort S]

**(a)** EklipZ `DangerAnalyzer.getFastestThreat` runs `dest_breadth_first_target` backward from our
general over enemy tiles. It looks for the *shortest* path along which an enemy sweep (collecting
army - 1 from every enemy tile it passes, paying our tiles' army) arrives with more than our general's
army. That catches trails of 3s and 4s and multi-tile threats that our single-cell
`A[i] - (d-1) + margin` misses.

The same routine from our side (our tiles → enemy general) is the "kill path":
- EklipZ `get_all_in_move` and `BotKillTiming`;
- it accounts for the trail of 1s becoming 2s if the land bonus hits mid-path (`all_in_strategy.md`).

**(b)** `DangerAnalyzer.getFastestThreat` L396, `SearchUtils.dest_breadth_first_target`;
`all_in_strategy.md` (army-bonus timing in the kill estimate).

**(c)** DP restricted to shortest-ish paths. It cannot loop, and costs O(n·depth):
```python
def sweep_values(self, target, owner, depth):
    """best[i] = max army an `owner` sweep starting at i arrives at `target` with,
    moving along cells with strictly decreasing distance to target."""
    dT = self.bfs([target])                 # cache per target
    best = [-INF] * self.n
    best[target] = 0
    order = sorted((i for i in range(self.n) if dT[i] <= depth), key=lambda i: dT[i])
    for i in order[1:]:
        own = self.O[i] == owner
        gain = (self.A[i] - 1) if own else -(self.A[i] + 1)   # neutral 0-cell costs 1 (left behind)
        bj = max((best[j] for j, _ in self.nb[i] if dT[j] == dT[i] - 1), default=-INF)
        if bj > -INF: best[i] = bj + gain
    return best, dT
```
- Enemy threat: `best, dT = sweep_values(g, 2, 14)`. For an enemy cell e,
  `surplus = best[e] - (A[g] + dT[e]//2)`. If it is > 0, that is a kill threat with eta = dT[e]. Feed
  `(e, eta, surplus)` into `garrison_need()` and 1.3.
- Our kill: `best, dT = sweep_values(egen, 1, 30)`, then
  `arrive = best[i] + (bonus_ticks_crossed(dT[i]) * (#own cells on path))`. Kill if
  `arrive > gen_est(dT[i]) + margin`. That replaces the top-4-stacks loop in `try_kill()`. It also
  finds "start at a 30-stack 4 tiles behind the front and sweep 3 more stacks on the way" kills.

Note: after turn 800 the deathtouch kill needs only a valid move onto the cell. Then
`need = 1 + (army of the cells on the path that must be beaten)`, and the plain path cost already does
that.

### 1.6 Expansion that maximises land per move ("capture rate")  [VALUE: HIGH, effort M]

**(a)** EklipZ's expansion (`ExpandUtils.get_round_plan_with_expansion` L193) does a **multiple-choice
knapsack over candidate capture paths**, with the turns left in the 50-tick cycle as the budget:
- each source tile proposes paths of several lengths (one choice per group);
- path value = Σ tile values: neutral 1.0, enemy 2.05 (`ENEMY_TILE_CAP_VALUE`), +0.01 undiscovered,
  plus tiny tie-breaks toward the centre or the enemy (`_get_tile_path_value` L2172);
- path cost = moves, *including* the moves through own land needed to reach the frontier;
- it runs in three phases:
  1. large-tile paths;
  2. small-tile paths;
  3. **leaf moves**: every frontier tile with army ≥ 2 next to a neutral captures it, 1 move, 1 tile
     (`_include_leaf_moves_in_exp_plan` L1666).

The design note says to use the 2s first after the land bonus (`MentalFramework.txt`: "use 2 tiles
first on round 2 before attacking").

Why: every capture move is +1 land. Every move through own land, or a gather move, is +0. hv measured
gather mode capturing on 9% of turns against 80% in expand mode. Juraj's land curve, the benchmark from
§2, is 38-41 land at t=100, 73-77 at t=200, 104 at t=300, 170 at t=600.

Our leaks:
- `path_to()` charges neutral 2 vs own 1, so launched stacks *avoid* capturing.
- `best_capture()` forbids stacks > small_cap from taking neutrals.
- Castle cycles burn up to 30 gather moves (`castle_gather_budget`).
- Attack cycles gather for 14 moves before every launch.

**(c)** Three cheap changes, then the planner:
```python
# 1. path_to(): when purpose in ('expand','scout') or target far, make captures cheaper than walking
#    own land:   own c = 1.0 ; neutral plain c = 0.9 ; enemy c = 1 + A[j]   (was own 1, neutral 2)
# 2. best_capture(): allow ANY non-general stack to take a neutral if no cycle reserved it.
# 3. Log capture_rate = captures / turns per 50-tick window (diagnostic; target >= 0.6 midgame).

def expansion_plan(self, max_plans=40):
    """Greedy multiple-choice knapsack: pick capture plans by value/moves until the cycle budget."""
    t = self.turn; left = 50 - (t % 50)
    used = set(); plans = []
    srcs = sorted((i for i in range(self.n) if self.O[i] == 1 and self.A[i] >= 2
                   and i not in self.reserved()), key=lambda i: -self.A[i])[:max_plans]
    dfront = self.bfs_frontier_own()          # distance via own cells to a cell adjacent to neutral
    for s in srcs:
        a = self.A[s] - (self.need_g if s == self.general else 0)
        w = dfront[s]                          # wasted moves to reach the frontier
        if a < 2 or w >= left: continue
        path = self.snake(s, steps=min(a - 1 - w, left - w), avoid=used)   # greedy DFS into neutral
        val = sum(2.05 if self.O[j] == 2 else 1.0 for j in path) + 0.01 * len(path)
        if path:
            plans.append((val / (len(path) + w), val, len(path) + w, s, path))
    plans.sort(reverse=True)
    budget = left; chosen = []
    for r, val, moves, s, path in plans:
        if moves <= budget and not used.intersection(path):
            chosen.append((r, s, path)); used.update(path); budget -= moves
    return chosen                               # first move of chosen[0] is the option

def snake(self, s, steps, avoid):
    """Walk from s (through own cells to the nearest frontier) then into neutral, each step
    choosing the neutral neighbour with the most neutral neighbours (avoid dead ends), and
    preferring cells toward the enemy (belief distance) on ties."""
```
Option scoring in `macro()`:
`w_expand * r` (r = captures per move, ≤ 1), with a deadline bonus while `t % 50 >= 50 - moves`.
This sits between garrison and launch. When the frontier has no neutral left (contact everywhere),
the plans collapse to enemy captures.

Combine with 1.1: if `en_gather_ratio` is low (the enemy is expanding), raise `w_expand` over
`w_cycle`.

### 1.7 Cycle structure: gather → launch → land before the bonus  [VALUE: MED, effort S]

**(a)** `BotModules/BotTimings.get_timings` L335 splits each 50-tick cycle:
- **gather** until `gatherSplit`;
- **launch** at `launchTiming = 50 - pathLen - 1` (+½·enemy/neutral tiles on the path, −½·friendly
  ones), clamped to ≤ 32;
- `gatherSplit = min(launchTiming, max(15, min(32, 50 - turns_of_good_expansion)))`, clamped to ≤ 24,
  with ±2 random jitter.

Effect: the main army arrives in enemy land in the last ~10-18 ticks of the cycle. Its captures (enemy
tile = 2.05) happen just before `t % 50 == 0`, so the enemy loses the bonus on those tiles. The first
part of the cycle (just after the bonus, when every tile is a fresh 2) is used for leaf expansion plus
gathering.

`calculate_greedy_turns_available` L53 then asks: "how many turns can we keep expanding before the
enemy's gathered fog army (1.1) exceeds the army on our defensive path?" That number sets how long the
gather phase is allowed to be greedy.

**(c)** In `new_cycle()`/`cycle_move()`, replace the fixed `gather_budget` with:
`budget = clamp(launch_tick - (t % 50), 4, 24)`, where
`launch_tick = clamp(50 - len(path_to(g, tgt)) - 1, gatherSplit, 32)`.
Do not start an attack cycle in the last `pathLen` ticks of a cycle; expand instead (1.6).

### 1.8 Enemy-general prediction: emergence field + launch-timing / land bounds  [VALUE: MED, effort S]

**(a)** Two layers.
- **Hard bounds** (`ArmyTracker.limit_gen_position_from_emergence` L3390):
  - `maxDist ≤ turn - launchTurn` and `maxDist ≤ tileCount - 1` (the territory is connected);
  - in the first 100 ticks, the first-launch trail reaches at most `launchTurn//2` tiles;
  - the enemy `launchTurn` is the tick at which their tile count first became 2.
- **Soft score** (`new_army_emerged` L1339): when an enemy army emerges at e, BFS from e **only
  through cells not visible last tick**, and add
  `max(1, plateau * s / max(plateau, dist))` to each fog cell, where
  - `plateau = clamp(dist_limit-1, 1, 5)`,
  - `s = 5*min(10, 2 + v**0.75) / (5 + t//25)` (big and early emergences are more informative).

The README calls this "insanely accurate".

**(c)** In `update()`:
```python
if self.prev and self.prev_opp_land == 1 and self.opp_land == 2: self.e_launch = self.turn - 1
for e in newly_seen_enemy_cells:                      # first_enemy_seen[e] == t
    self.first_enemy_land[e] = self.opp_land + self.enemy_tiles_lost_so_far
# prune_candidates(): add the connectivity bound (currently only the time bound is applied)
if dc[e] > self.first_enemy_land[e] - 1: ok = False
if self.turn <= 60 and self.e_launch is not None and t_first_seen(e) <= self.e_launch + self.e_launch//2:
    if dc[e] > self.e_launch // 2 + 1: ok = False      # first trail cannot reach farther
# belief_scores(): replace `de` (bfs over all cells from recent enemy land) with the emergence field
self.emerg[c] += s * plateau / max(plateau, d_fog(e, c))   # d_fog = bfs through non-visible cells
sc -= w_emerg * log1p(self.emerg[c])
```
`first_enemy_land` is a strict and cheap bound. At t≈60 the enemy has ~25 land, so every seen enemy cell
pins the general within 24 steps, against the current time bound of ~60.

### 1.9 All-in rules (when to go for the general)  [VALUE: MED, effort S]

EklipZ has two triggers (`BotCombatOps.check_should_be_all_in_losing` L1659,
`determine_should_winning_all_in` L1880):
- **Winning all-in:**
  - condition: `ourStanding ≥ 100` and `ourStanding > 2·oppStanding + pathLen`, with hysteresis
    `> 1.4·oppStanding + pathLen/2` once on; standing = army − land;
  - action: stop expansion and castles, set a 50-tick (30 if our land − pathLen < 60) all-in cycle,
    and gather everything (a prize-collecting Steiner gather) onto the path to the predicted general
    and its top-3 emergence cells, hitting in `hitGeneralInTurns`.
- **Losing all-in:**
  - econ = land + 35·castles;
  - `econ_en > 1.05·econ_us + 10` (1.08 and +1 castle early in a cycle) for
    `> max(50, land/5 + 15)` consecutive ticks;
  - or immediately if t > 250 and `econ_en > 1.3·econ_us + 5 + 20·(castles+2)` and
    `standing_en > 1.25·standing_us + 5`;
  - with a counter +3/tick if t > 150 and the ratios are 1.4 / 1.25.
- **Projected round loss** (`all_in_strategy.md`): if we are projected to lose the round by >10%,
  check an immediate rally straight into the general, counting the +1 bonus on the path trail if impact
  lands after `t % 50 == 0`.

Adapted to this ruleset (deathtouch at 800, draw at 1200):
```python
def all_in_mode(self):
    st_us = self.my_army - self.my_land; st_en = self.opp_army - self.opp_land
    econ_us = self.my_land + 35*len(self.my_castles); econ_en = self.opp_land + 35*len(self.enemy_castles)
    D = self.dist_to_belief()
    win = st_us >= 100 and st_us > (1.4 if self.allin == 'win' else 2.0)*st_en + D*(0.5 if self.allin else 1)
    lose_tick = econ_en > 1.05*econ_us + 10
    self.lose_ctr = self.lose_ctr + 1 if lose_tick else 0
    lose = self.lose_ctr > max(50, self.my_land//5 + 15) and self.turn < 760
    # after 800 a lost economy is not lost: switch to the deathtouch swarm instead (§2.6)
    return 'win' if win else 'lose' if lose else None
```
In 'win' mode, `new_cycle()` uses root = g, purpose='attack', budget = time to the next cycle-end
minus D, with no castle cycles. In 'lose' mode, also stop `home_fill`/scout and allow
`garrison_cap_frac = 0.2`.

### 1.10 Gather tree: prune to max value-per-turn  [VALUE: LOW-MED, effort S]
`Gather/GatherPrune.prune_mst_to_max_army_per_turn_with_values` L620 builds the full gather tree
(MST/knapsack), then repeatedly prunes the leaf with the lowest `value/gatherTurns` (tie
`trunkValue/trunkDistance`). It stops when pruning would lower the tree's average army per turn or drop
below `minArmy`. Castles and the general on the tree add +1 per 2 remaining turns
(`cityCounter * turnsLeft//2`).

Ours is a greedy forward selection without the pruning pass and without counting castle/general growth
during the gather. Add a backward pass: after `gather_select`, drop selected leaves whose
`(A-1)/depth_increment` is below `total/used` while total stays ≥ need. Cheap, but it only improves
gather efficiency a little.

---------------------------------------------------------------------------------------------------

## 2. relh Sentinel (built for THIS ruleset) + juraj C++ (V3.4/V3.5): extracted algorithms

Sources:
- `vendor/ext/relh_generals-bots/generals/agents/sentinel_agent.py` (v2) and `sentinel_v3..v21_agent.py`,
  plus `docs/agent-development/*`;
- `vendor/ext/juraj/juraj_v35/VALIDATION_REPORT.md`, `juraj_v35/main.cpp`, `core.hpp`;
- `juraj/juraj_v34/*.inc` (split parts included by main.cpp).

Measured strength (their own reports):
- **Sentinel v2:**
  - 97.7% vs Expander and 96.1% vs Hunter (256 games each);
  - 46.9-62.5% vs the Amin PPO;
  - 11-5 vs Juraj V3.4 and 2-14 vs V3.5.
  - No version from v3 to v21 held up as an improvement, except **v4** (a 2-step threat check before a
    castle build: 13-2-1 vs V3.4).
- **Juraj V3.5:** 53W/6D/41L in a 100-game run. Its land curve (mean, alive games) is a **benchmark for
  us**:

| Turn | Land |
|---|---|
| 50 | 17 |
| 100 | 38-41 |
| 150 | 56-59 |
| 200 | 73-77 |
| 300 | 104 |
| 400 | 126 |
| 600 | 170 |
| 800 | 207 |

- Juraj's loss causes:
  - "defense undercommitted" 14;
  - "early stall" 11-17 (land100 < 30 and > 5% passes);
  - "interceptor not used" 10 (a stack adjacent to the attacker could have killed it, often with 30
    ticks of warning);
  - false emergency defense (`distance <= 5` predicate): 200-600 defense actions and reversals per
    game; it was fixed by requiring a strong enemy within 2, or a tracked *moving* stack within 5.
- **relh v3**, which added fog memory of enemy armies, was 8pp WORSE: a 47-army sighting stayed
  "approaching" after a battle had consumed it. **Lesson for 1.2:** subtract fight losses from tracks.
- **relh v17 audit:** a 247-stack first seen 11 tiles from home could not be stopped by any gather in
  time. Defense must be pre-positioned, which argues for 1.1, 1.4 and the corridor below.

### 2.1 Chase-kill the attacker from an adjacent stack  [HIGH, effort S, ~40 lines]
(a) If an enemy stack qualifies as a threat and one of our non-general cells x is adjacent to it with
A[x] - 1 > A[e], hit it now. This is Juraj's "interceptor not used" fix (score 9000 - A[x]).

(b) `juraj_v35/main.cpp:46,~55`; `v34_part04.inc:78 defense_emergency`; relh
`sentinel_agent.py:197 removed_threat` (attack from a third tile, never head-on with the general).

(c) New `Bot.intercept()`, called in `decide()` right after `urgent_defense`. Keep `self.pO, self.pA`
from the previous tick.
```python
def qualified_threats(self):
    g = self.general; Ag = self.A[g]; out = []
    for e in range(self.n):
        if self.O[e] != 2 or self.A[e] < 2: continue
        eta = self.dist_g[e]; a = self.A[e]
        moving = self.pO is not None and ((self.pO[e] == 2 and self.pA[e] != a) or any(
            self.pO[z] == 2 and self.pA[z] >= a - 2 and self.pA[e] <= 1 for z, _ in self.nb[e]))
        lethal = eta == 1 and (self.turn >= 800 or a - 1 > Ag)
        if lethal or (eta <= 2 and a >= max(4, Ag // 2)) or (moving and eta <= 5 and a >= 6):
            out.append((eta, -a, e))
    return [e for _, _, e in sorted(out)]

def intercept(self):
    best = None
    for e in self.qualified_threats():
        for x, _ in self.nb[e]:
            if self.O[x] == 1 and x != self.general and self.A[x] - 1 > self.A[e]:
                k = -self.A[x]                          # smallest stack that still wins
                if best is None or k > best[0]: best = (k, x, e)
    if best:
        a = self.mv(best[1], best[2]); return a if self.safe(a) else None
```
Juraj's attack-safety check (main.cpp:32) goes into `best_capture` too: after capturing j, reject the
move if another enemy adjacent to j has ≥ the army we would leave there.

### 2.2 Block at the attacker's next choke, unless we win the race  [HIGH, effort M]
(a) The choke is a cell on the attacker's shortest-path set. Any of our stacks that reaches it strictly
before the attacker (dx[c] < de[c]) and has more army walks there (score 8000 - A[x]). Otherwise, if
the attacker is ≤ 3 away, stacks closer to the general reinforce it.

**Race exemption:** skip defense if one of our stacks reaches the enemy general sooner
(dist + 1 < threat eta) with more than its army plus the distance.

This is the cheap version of 1.3. Implement 2.2 first; 1.3's gather-with-deadline is the upgrade.

(b) `juraj_v35/main.cpp:56-57`, `v34_part04.inc:78`.

(c)
```python
e = threats[0]; de = self.bfs([e]); eta = de[self.general]
onpath = [c for c in range(self.n) if de[c] + self.dist_g[c] == eta]
for x in top3_own_stacks(excluding g) with A[x] - 1 > A[e] + 1:
    dx = self.bfs([x])
    c = min((c for c in onpath if dx[c] < de[c]), key=lambda c: (dx[c], -de[c]), default=None)
    if c is not None: return self.mv(x, self.path_to(x, c)[1])
```

### 2.3 Approach-threat state machine for tracks (fixes stale memory and over-defense)  [HIGH, effort M]
(a) Juraj tracks the *most important* visible stack, with importance = a·(1 + max(0, 18-d)/18).
- **Same stack** if its travel ≤ dt + 2 and the army ratio is within 2×. Closing rate =
  (prev_dist − dist)/dt.
- **Threat score** = 0.55·old + 0.45·new, where new is a weighted sum:

| Weight | Term |
|---|---|
| 0.25 | army vs our reaction stack (capped at 1) |
| 0.20 | share of opp_army |
| 0.15 | proximity |
| 0.20 | closing rate |
| 0.10 | persistence |
| 0.10 | our gap |

- **Activates** when seen 2 ticks in a row, closing, army ≥ 8, ≥ 12% of opp_army and score ≥ 0.55.
- **Clears** after 5 ticks at ≤ 0.30.
- **Unseen decay:** ×0.90 (≤ 3 ticks), ×0.72 (≤ 7), then ×0.45.
- **While active:**
  - the reaction stack stays home (no outward moves, no half splits);
  - the reaction counts as adequate if its army ≥ the attacker's and it is ≤ 2 tiles farther from home
    than the attacker;
  - castle builds are delayed.

(b) `juraj_v34/v34_part03.inc:66 update_approach_threat`; `v34_part05.inc:43`.

(c) Combine with 1.2:
- keep `track.score`, `track.closing`, `track.active`;
- `garrison_need()` only uses *active* tracks not already answered by an adequate reaction stack;
- subtract fight losses: when `built == 0` and our accounting's R shows an enemy army drop
  (`loss = max(0, -(R - expected_struct_growth))`), take `loss` off the track nearest the fight cell
  (the cell that changed owner) and drop the track at ≤ 0. This is the relh v3 failure mode.

### 2.4 Required-defense formula (replaces the visible loop in `garrison_need`)  [MED-HIGH, effort S]
Juraj V3.4 (`v34_part03.inc:110`). Let s be the largest visible enemy stack at distance dg, and
`local` = A[g] + Σ(A[j] - 1 for own j adjacent to g).
- **dg ≤ 5:** `need = s + max(3, (8 - dg)//2) + 4`.
- **dg ≤ 8:** the same `need`, but only if `s + 3 ≥ local - 2`, or it closed in on 2 sightings and
  `s ≥ max(8, local//2)`.
- **dg > 8:** watch only.

Ours (`A[i] - (d-1) + 2` for d < 10) over-reserves at d = 6-10. relh: trigger the recall on the need
without the +6 buffer, and use the buffer only to veto moving army *off* the general.

Per-cell reserves (v35 `main.cpp:28`):
- general: max(5, opp_army//12);
- a cell adjacent to an enemy: max(2, that enemy's army + 1);
- any other cell: 1.

### 2.5 Castle plan for this ruleset  [HIGH, effort M]
Evidence across three repos says **castles are a bet on the game lasting; build opportunistically,
never by cadence**:
- hv: ~15 builds/game was -36 Elo head-to-head. Opportunistic builds (0.9-1.2/game, only where a stack
  already stands on a 35-cost site) were +111 Elo over never building. Raising the "walk army to a
  site" threshold from 75 to 400 total army stopped gather mode eating 72% of the midgame, recovered
  ~30 land by t=200 and gained +160 Elo (`hv-nguyeen_Generals-RL-bot/bot/config.py:115-134`).
- bca (blake-ar): the policy almost never builds. Forcing builds is harmful on average; the useful
  subset is t ≤ 700, behind on army or land, post-build garrison ≥ 10, and enemy > 3 away
  (`generals/training/counterfactual.py:766-772`).

Our `n_castles_wanted()` = 1 + (t-90)/45 (≈14 by t=720), with `castle_gather_budget=30`, is exactly
the measured-losing pattern. It is a prime suspect for "out-economied on land".

Rules to port:
- **Juraj sites** (`v34_part02.inc:88`), computed at t=0:
  - 3 sites ≥ 5 from the general, at price 35 (≥ 7 manhattan from the general and the other sites);
  - maximise traffic toward the likely spawns (weighted by our prior), +120 for a different exit from
    the general, +80 for a new sector, plus small choke/degree/room bonuses.
  - Result: castles on the approach lanes, which give income plus a forward blocker and staging point.
- **Juraj windows:** C1 130-175, C2 225-285, C3 300-375. Funding starts at t=90, then 45 ticks after
  each build. **Just-in-time funding:** latest start = deadline − (Σ feeder distances + 1) − 5.
  **Always use the live price**, which caused V3.5's illegal builds.
- **Build guards:**
  - site army ≥ live price;
  - (our army − price) ≥ max(20, opp_army/2);
  - no visible enemy stack within 2;
  - approach threat (2.3) inactive.
- **relh v4 guard (+15.6pp):** site army ≥ price + 5 + (biggest enemy stack that can reach the site
  in 2 moves); t + 2·price + 100 < 1200.
- **relh score:** 45 + 0.08·(1200 − t) − 0.3·price.

(c) For our Bot, in `castle_build_now()`:
```python
ok = (price == 35 or (behind and price <= 41)) and A[i] >= price + 5 + reach2(i) \
     and (self.my_army - price) >= max(20, self.opp_army // 2) and not approach_active \
     and t + 2*price + 100 < 1200
# score: 45 + 0.08*(1200 - t) - 0.3*price ; build only from a stack already standing there
```
In `new_cycle()`, castle-purpose gather cycles are allowed only when
`(behind or my_army - my_land > 400)` and `threat_eta > 12`, capped at 3-4 castles when ahead.
`castle_site()` prefers precomputed lane sites (Juraj) over "near home". Our current preference for 2
castles within 3 of home costs 43-47 each; worth an A/B test against lane sites.

Before tuning, check how many castles the neural clone builds per game (arena logs). If it builds
many, that is a style to *punish* (it has less mobile army), not to copy.

### 2.6 All-in / kill pricing and the deathtouch endgame  [MED-HIGH, effort S]
- **Juraj attack test** (`v34_part03.inc:77`): with `seen` = last-seen general army and
  `hidden` = the rest of opp_army,
  `expected = seen + hidden / max(3, d+2)`; attack if `stack - 1 > expected + max(2, d//3)` or
  t ≥ 720.
- **hv** (`controller.py:384-431`):
  - general visible: `est = A[eg] + Σ enemy within d//2 of eg + d//2`, margin 1.25;
  - general inferred: `est = max(last_army + age//2, 0.2·(opp_army − visible army farther than d))`,
    margin 1.25 if ≤ 4 candidates else 2.0, and no unlocated commit before t=170.
  - A big enemy stack seen far from its general means the general is exposed: counter-strike. This
    pairs with 1.1.
- **Two-tile pincer** (`juraj_v35/main.cpp:~54`): a stack next to an enemy castle or general that
  cannot take it alone is fed by a neighbour when (combined − 2) > target + 1 (score 8500, +400 for the
  general). Add it to `win_now()/try_kill()`.
- **Deathtouch defense** (relh `sentinel_agent.py:218`): after 800, any adjacent enemy with ≥ 2 wins on
  its move, so kill the source from a third tile; the move order gives that priority.
  - Our `resolve()` models this. The gap is attackers 2 away: from ~760 keep every cell at distance 2
    from the general owned with ≥ 2. This extends `endgame()`, which only secures distance 1.
  - On offense, from t ≥ 800 give +2000 to moves that end adjacent to the enemy general when we have
    ≥ 2 stacks adjacent (Juraj).

### 2.7 Land-deficit modes, tick-graded capture bonus, anti-stall, anti-oscillation  [MED, effort S]
- **Deficit modes:**
  - Juraj V2.7.2 emergency: land < 0.65·theirs, or < 0.75 while they gained 3+ more over 20 ticks;
    it ends at ≥ 0.85.
  - V3.5: "severe" when they lead by > 18 land (expansion share 62%, no scouting, war halved); "soft"
    when they lead by > 8.
- **Tick-graded neutral bonus** (`v34_part04.inc:171`): +220 within 5 ticks of the land tick, +150
  within 10, +90 within 20, plus +150/+200/+400 when behind / in deficit / severe. Port it as a graded
  `bonus_mult` in `best_capture()`.
- **Anti-stall:** never PASS while a safe neutral capture exists (our idle path → capture).
- **Anti-oscillation** (`juraj_v35/core.hpp route_allowed`): a stack may not step back onto its last
  2-4 cells unless an event happened since (first contact, a new threat, the general found, a castle
  lost). relh found reversals were its main failure from v6 on. Add a 12-move history to `cycle_move`.

### 2.8 Pull rear surplus forward once the general is found  [MED, effort M]
Source: `juraj_v34/v34_part06.inc:2`.
- Once egen is known, move surplus toward a staging cell 2-4 tiles from the target, keeping 1 per cell
  (3 on castles).
- Use only "safe rear" cells (edges, dead ends, off-route), skipping any within 5 of an enemy stack
  of 8+.
- Rank by surplus × progress, and commit at least half of the surplus that existed when the general
  was found.
- In our Bot this is a second gather root for `new_cycle()` when `egen >= 0`.

---------------------------------------------------------------------------------------------------

## 3. RL bots (Amin, blake-ar/bca, quant-eagle, hv-nguyeen; Klincent is an empty checkout)

### 3.1 Reward shaping
- The two strongest RL entrants used **sparse terminal reward only**: bca at ladder #3
  (`blake-ar_generals-bots/generals/training/README.md:99`) and quant-eagle at #10
  (`docs/DESIGN.md`).
- Amin's shaped rewards produced a weak bot (rank ~105/116):
  - land/army ratio deltas;
  - castle count;
  - a contact-centroid potential;
  - a general-safety potential.
- hv: land shaping has zero expectation in mirror self-play (`docs/ml-log.md`).
- **Nothing to port from the shaping.**

### 3.2 Features that mattered
The nets only worked once they got memory and geodesic channels:
- **bca** (`competition/agents/conv_1313/bot.py:353-442`):
  - last-seen enemy army plus log(age);
  - 7 frames of army deltas;
  - a 512-turn opp_army/opp_land history;
  - turn % 50 phase, a build-cost plane, a deathtouch flag and countdown;
  - inferred castles.
- **quant-eagle** (`docs/DESIGN.md:63-90`):
  - multi-scale scoreboard rings;
  - army EMAs at α 1/2..1/128;
  - BFS planes to the remembered enemy general, our general and the frontier;
  - auxiliary heads for "where is the enemy general" and "we die within 32 ticks".
- **hv:** the observation was the ceiling; no net imitated its heuristic until belief channels were
  added (`ml-log.md:767`).

**The scalar history (opp_land and opp_army over time) is what all of them leaned on.** This confirms
1.1.

### 3.3 Guards: measured values for our safety filter  [HIGH, effort S]
hv ladder forensics (`bot/policy/guard.py:10-20, 95-130`):
- 16 of 50 losses had army leave the general in the last 15 ticks with an enemy stack within 3.
- When the general was emptied, the visible threat within 3 was ≈ 0 in 9 of 10 losses. The killer
  arrived 7-36 ticks later.
- Hidden > garrison held in 10 of 10 losses (median 191 vs 38), yet it is **useless as a trigger**:

| Trigger | Measured cost |
|---|---|
| Veto if hidden ≥ 1.0× garrison | −322 Elo (fires 98% of turns) |
| Veto if hidden ≥ 4.0× garrison | −61 Elo |
| Lock the general whenever anything is within 6 | −265 Elo |
| Standing garrison of 5% of enemy army (`config.py:157`) | harmful even at 0.05 |

- **The only safe veto:** don't move army off the general when a visible enemy within 3 has
  ≥ 0.5 × the garrison.

This confirms that 1.1 must be *better information* (gather-move counting), not a bigger hidden-army
multiplier.

(c) In `safe()` and the root == g launch branch of `cycle_move()`:
```python
def drain_ok(self, a):
    if a[0] != 0 or a[1]*self.W + a[2] != self.general: return True
    near = max([self.A[i] for i in range(self.n) if self.O[i] == 2 and self.dist_g[i] <= 3] +
               [arm for c, arm, ts, *_ in self.tracks if self.dist_g[c] - (self.turn - ts) <= 3] + [0])
    return near < 0.5 * self.A[self.general]
```

### 3.4 Enemy fog-activity counter (the cheap version of 1.1)  [HIGH, effort S]
Each tick, idle = (Δopp_land + enemy tiles we took ≤ 0) and no build and no visible enemy change. A
run of idle ticks with nothing visible is the signature of a stack forming in fog.
```python
self.e_idle.append(idle)          # deque(maxlen=40)
G20 = sum(list(self.e_idle)[-20:])
# garrison_need() hidden branch:
f *= min(1.5, max(0.4, 0.4 + G20 / 12.0))
ni = int(min(hidden, self.est_gen_army() + G20 * max(1, hidden // max(1, self.opp_land))) * f)
```
Also add G20, G40 and the 50-tick opp_army slope (minus known growth) to `threat_features()`. Then
retrain `learn/train_threat.py`, ideally with quant-eagle's "die within 32 ticks" target, and turn on
`learned_threat_w`.

### 3.5 Style facts about strong ladder bots (useful constraints)
- Top players split on 0-2% of turns (quant-eagle `train/magnet.py:51`). Count our split rate and
  prefer full moves from non-general tiles.
- Dominant kill pattern: expand to parity, build one big stack, commit it blind along a geodesic into
  enemy land, keep the general hidden, don't churn within 3 of home.
- Rank-1 Kubic: capture rate 0.39, 117 idle turns, typical strike stack 47 (`hv config.py:200-206`).
- Launch thresholds: max(20, 0.10·my_army) rather than our `min_stack=6`. hv: 8 / 0.04 lost 241 Elo on
  the ladder although it was +99 locally.
- hv's losses are lopsided only before t=200 (2.25:1); later bands are coin flips. **Early defense and
  economy decide games.**
- Measurement warning (hv): local wins against one fixed opponent repeatedly inverted on the ladder, so
  gauge changes against a diverse pool.

---------------------------------------------------------------------------------------------------

## 4. RANKED PORTING LIST (value / effort). Implement top-down and A/B each against a diverse pool

| # | Idea | Section | Value | Effort | Replaces / extends |
|---|------|---------|-------|--------|--------------------|
| 1 | Chase-kill: an adjacent stack kills a qualified attacker (never with the general) | 2.1 | HIGH | S | new `intercept()` after `urgent_defense` in `decide()` |
| 2 | Castle gating: drop the 45-tick cadence; build only from a stack already standing on a 35-cost site, with the 2-move-threat guard, army-after ≥ max(20, opp/2), and t+2·price+100<1200; castle gather cycles only when behind or very rich | 2.5 | HIGH | S | `n_castles_wanted`, `new_cycle`, `castle_build_now` |
| 3 | Land-per-move leaks: in `path_to` neutral costs 0.9 vs own 1.0 for launch/scout; big stacks may take neutrals; tick-graded neutral bonus; never pass with a safe capture; log capture rate | 1.6, 2.7 | HIGH | S | `path_to`, `best_capture`, `macro` idle path |
| 4 | Enemy move classifier (Δopp_land, visible deltas, builds) → G20 idle counter, then the full fog gather-queue stack estimate `fog_risk()` | 3.4 → 1.1 | HIGH | S→M | `update()`, the hidden branch of `garrison_need()` |
| 5 | Fog tracks advance only on unexplained enemy moves, follow the expected path, and lose army in fights (fixes relh v3's stale memory) | 1.2, 2.3 | HIGH | S-M | `update_tracks`, track distance in `garrison_need` |
| 6 | Narrow general-drain veto: no move off the general when a visible or tracked enemy within 3 has ≥ 0.5·A[g] (the only safe veto hv measured) | 3.3 | HIGH | S | `safe()`, the root==g launch branch of `cycle_move` |
| 7 | Defend at the attacker's path/choke: block if we arrive first, else reinforce the general, with the race exemption; upgrade to multi-root deadline gathers on the threat path | 2.2 → 1.3, 1.4 | HIGH | M | garrison option in `macro()` |
| 8 | Sweep DP (army collected along shortest paths) for enemy kill threats and for our kill check, plus Juraj/hv kill pricing (hidden/(d+2), margin 1.25/2.0) and the pincer | 1.5, 2.6 | MED-HIGH | S | `try_kill`, `win_now`, `garrison_need` |
| 9 | Juraj required-defense formula (≤5: s+max(3,(8−d)/2)+4; ≤8 only if dangerous) instead of `A−(d−1)+2` out to 10 | 2.4 | MED-HIGH | S | visible loop in `garrison_need` |
| 10 | General belief: the connectivity bound dist ≤ opp_land_at_sighting−1, launch-trail bound, emergence field through fog | 1.8 | MED | S | `prune_candidates`, `belief_scores` |
| 11 | Full expansion planner: greedy multiple-choice knapsack of snake paths by captures/move within the cycle | 1.6 | HIGH | M | new option in `macro()` |
| 12 | All-in triggers: win (standing ≥100 and >2× +D) and lose (econ deficit counter); deathtouch distance-2 ring from 760 | 1.9, 2.6 | MED | S | `macro()` mode flag, `endgame()` |
| 13 | Cycle timing: launch at 50−pathLen−1 (≤32), gather budget = launch_tick − t%50 | 1.7 | MED | S | `new_cycle` budget |
| 14 | Anti-oscillation route memory for stacks | 2.7 | MED | S | `cycle_move` |
| 15 | Pull safe-rear surplus forward once egen is known; lane castle sites (Juraj) | 2.8, 2.5 | MED | M | `new_cycle`, `castle_site`, `setup` |
| 16 | Gather prune pass (value/turn) + castle/general growth during gather | 1.10 | LOW-MED | S | `gather_select` |

Cross-cutting evidence:
- **Defense:** both of our failure modes are documented in other bots.
  - Over-defense / hoarding loses land: Juraj's false emergencies, hv's −265 / −322 Elo vetoes.
  - Under-defense against a stack built in fog: Juraj's 14 undercommitted, hv's 16/50 drained-general
    losses, relh's 247-stack.
  - The fix in every case is **better information** (move counting, tracks with move budgets, chokes),
    not bigger multipliers.
- **Economy:** "fewer land mid-game" is mostly *capture rate* (moves spent not capturing). Cadence
  castles and gather cycles are the measured culprits (hv: gather mode captures 6-9% vs 80% for
  expand).
- **Benchmark:** Juraj land at t=100/200/300/600 = 40/75/104/170. Compare our bot's land curve against
  it (arena logs) before and after #2/#3.

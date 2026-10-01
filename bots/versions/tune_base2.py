"""
Code Bot entry — participant: [PARTICIPANT_ID]   bot: [BOT_NAME]

Strategy (heuristic + exact rules):
  * Exact bookkeeping of the opponent from opp_army / opp_land each turn: every
    enemy castle build is detected with its exact price; new type-5 cells (after
    turn 0 all mountains are known) locate the castle; the price pins the enemy
    general to a Manhattan ring around it.
  * Enemy-general belief from the map generator's spawn rules (BFS >= 17,
    |room7 - ours| <= 5), fog observations, castle-price rings and land counts.
  * Wave opening (~25 land at turn 50), land-bonus expansion, castle economy from
    the general's stack, threat-scaled garrison, gather trees (pruned by value per
    move), stack strikes at the general, deathtouch fortress + strike after 800.
  * One-turn exact resolution (pinned move order: chasing > reinforcing >
    smaller army) to veto moves that lose the general.

Sources / attribution:
  * Rules and move-order semantics re-implemented from strakam/generals-bots
    @13db8f69 (MIT), which is the pinned competition engine.
  * Ideas (not code): EklipZ generals-bot gather pruning / back-tracing;
    Straka & Schmid, "Artificial Generals Intelligence" (arXiv 2507.06825);
    statistics from public generals.bot tournament replays.
AI assistance: developed with Claude Code (Anthropic) as a coding assistant;
all strategy code is in this file and was reviewed by the participant.
Standard library only.
"""
import gc
import time
from collections import deque
from heapq import heappush, heappop

PARAMS = {
    # opening
    "open_div": 2,            # launch when (a-1) >= (49 - t - transit) // open_div
    "open_end": 50,
    "open_plan_s": 1.5,
    # garrison / defense
    "garrison_min": 2,
    "garrison_frac_hidden": 0.5,    # fraction of largest possible hidden stack kept home
    "garrison_cap_frac": 0.5,      # never keep more than this fraction of our army at home
    "threat_margin": 2,
    "threat_vis_range": 10,
    "threat_decay": 1.0,
    # option weights
    "w_garrison": 6.0,
    "w_garrison_urgent": 9.0,
    "hidden_stack_frac": 0.5,
    "track_min": 8,
    "track_frac": 0.12,
    "track_ttl": 30,
    "track_threat_dist": 12,
    "w_build": 7.0,
    "w_cycle": 1.2,
    "w_launch": 3.2,
    "w_scout": 0.9,
    "attack_root_front": 0,
    "scout_start": 60,
    "scout_min": 3,
    "scout_max_frac": 0.15,
    "regather_budget": 8,
    # captures
    "v_neutral": 1.0,
    "v_enemy": 2.2,
    "v_kill": 0.08,
    "v_ecastle": 8.0,
    "v_near_home": 2.0,
    "bonus_mult": 2.0,
    "bonus_window": 12,
    "bonus_lead": 0,
    "toward_w": 0.3,
    "home_r": 5,
    "v_home": 1.5,
    "w_home_fill": 2.4,
    "home_fill_start": 50,
    "small_min": 3,
    "small_frac": 0.04,
    "gen_move_pen": 0.6,
    # castles
    "castle_start": 90,
    "castle_stop": 720,
    "castle_every": 45,
    "castle_horizon": 900,
    "castle_reserve": 1,
    "castle_safe_dist": 4,
    "castle_val_min": 10.0,
    "castle_move_w": 1.0,
    "castle_price_w": 1.0,
    "castle_safety_w": 0.5,
    "castle_gather_budget": 30,
    "castle_home_n": 2,
    "castle_home_maxd": 3,
    "castle_home_w": 6.0,
    # army cycle
    "gather_budget": 14,
    "gather_default_budget": 40,
    "launch_max": 40,
    "min_stack": 6,
    "feed_min": 6,
    "kill_margin": 2,
    "intercept_dist": 4,
    "belief_enemy_w": 0.7,
    "belief_explore_w": 1.0,
    "belief_prior_w": 3.0,
    "attack_min_army": 60,
    # time
    "soft_budget_ms": 45,
    "first_budget_ms": 3000,
    # endgame
    "fortress_turn": 740,
    "dt_stage_turn": 760,
    "aggro_turn": 1000,
    "expand_toward_w": 0.15,
}

# learned spawn prior (logistic regression on generator samples; see learn/train_prior.py)
PRIOR_W = [-0.2466, 0.1374, 0.0267, 0.6231, 1.4779, 0.2212, -0.1133, 0.0064, -0.049, 13.246]
PRIOR_B = -5.1142

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PASS = [1, 0, 0, 0, 0]
INF = 10 ** 9
DEBUG = False  # arena sets True to surface exceptions


def _now():
    return time.perf_counter()


class Bot:
    def __init__(self, obs):
        self.H = H = int(obs["height"])
        self.W = W = int(obs["width"])
        self.pid = int(obs.get("player_id", 0))
        self.n = n = H * W
        nb = []
        for i in range(n):
            r, c = divmod(i, W)
            lst = []
            for d, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    lst.append((rr * W + cc, d))
            nb.append(lst)
        self.nb = nb
        self.mountain = [False] * n
        self.castle = [False] * n          # known castle cells (any owner)
        self.last_owner = [0] * n          # 0 unknown/neutral, 1 me, 2 enemy
        self.last_army = [0] * n
        self.last_seen = [-1] * n
        self.first_enemy_seen = [-1] * n
        self.ever_visible = [False] * n
        self.general = -1
        self.egen = -1
        self.prev = None                   # previous turn scalars and structures
        self.last_action = PASS
        self.enemy_castles = set()
        self.enemy_struct = 1              # enemy general + castles
        self.enemy_builds = []             # (turn, cell, price)
        self.rings = []                    # (cell, exact_d or None for >=7)
        self.plan = None
        self.cyc = None
        self.open_cfg = {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}
        self.open_dc = None
        self.t_start = time.perf_counter()
        self.budget_s = PARAMS["soft_budget_ms"] / 1000.0
        self.last_label = ""
        self.pending_stack = None
        self.threat_eta = INF
        self.tracks = []
        self.last_purpose = None
        self.need_g = 2
        self.turn = -1
        self.err = 0
        self.cands = None
        self.cand_dist = {}

    # ------------------------------------------------------------------ utils
    def late(self, frac=1.0):
        """True once the soft per-move time budget is (frac) used."""
        return time.perf_counter() > self.t_start + frac * self.budget_s

    def manh(self, a, b):
        W = self.W
        return abs(a // W - b // W) + abs(a % W - b % W)

    def bfs(self, sources, passable=None, limit=INF):
        dist = [INF] * self.n
        dq = deque()
        for s in sources:
            dist[s] = 0
            dq.append(s)
        nb = self.nb
        pas = self.pas if passable is None else passable
        while dq:
            i = dq.popleft()
            nd = dist[i] + 1
            if nd > limit:
                continue
            for j, _ in nb[i]:
                if pas[j] and dist[j] > nd:
                    dist[j] = nd
                    dq.append(j)
        return dist

    def dir_to(self, i, j):
        for k, d in self.nb[i]:
            if k == j:
                return d
        return -1

    def mv(self, i, j, split=0):
        d = self.dir_to(i, j)
        if d < 0:
            return None
        return [0, i // self.W, i % self.W, d, split]

    # --------------------------------------------------------------- parsing
    def parse(self, obs):
        H, W = self.H, self.W

        def flat(g):
            if g is None:
                return [0] * (H * W)
            if hasattr(g, "tolist"):
                g = g.tolist()
            if len(g) == H * W and not isinstance(g[0], (list, tuple)):
                return [int(v) for v in g]
            out = []
            for row in g:
                out.extend(int(v) for v in row)
            return out

        return flat(obs["type"]), flat(obs["owner"]), flat(obs["army"])

    # ------------------------------------------------------------- turn zero
    def setup(self, T):
        n = self.n
        for i in range(n):
            if T[i] in (2, 5):
                self.mountain[i] = True
        self.pas = [not m for m in self.mountain]
        g = self.general
        self.dist_g = self.bfs([g])
        # room7 for every passable cell (cells reachable within 7, excluding self)
        room = [0] * n
        for i in range(n):
            if self.pas[i]:
                d = self.bfs([i], limit=7)
                room[i] = sum(1 for v in d if v <= 7) - 1
        self.room = room
        mine = room[g]
        far = [i for i in range(n) if self.pas[i] and self.dist_g[i] < INF and self.dist_g[i] >= 17]
        cands = [i for i in far if abs(room[i] - mine) <= 5]
        if not cands:
            if far:
                best = min(abs(room[i] - mine) for i in far)
                cands = [i for i in far if abs(room[i] - mine) == best]
            else:
                mx = max(v for v in self.dist_g if v < INF)
                cands = [i for i in range(n) if self.dist_g[i] == mx]
        self.cands0 = list(cands)
        self.cands = set(cands)
        for c in cands:
            self.cand_dist[c] = self.bfs([c])
        self.prior = {}
        try:
            feats = {c: self.cand_features(c) for c in cands}
            for c, f in feats.items():
                self.prior[c] = sum(w * x for w, x in zip(PRIOR_W, f)) + PRIOR_B
        except Exception:
            self.prior = {c: 0.0 for c in cands}
        try:
            self.plan_opening()
        except Exception:
            if DEBUG:
                raise
            self.open_cfg = {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}

    def cand_features(self, c):
        """Static features of a spawn candidate (relative to our general)."""
        H, W = self.H, self.W
        g = self.general
        dc = self.cand_dist[c]
        reach = [v for v in dc if v < INF]
        span = max(reach)
        ecc_g = max(v for v in self.dist_g if v < INF)
        r, col = divmod(c, W)
        edge = min(r, H - 1 - r, col, W - 1 - col)
        nb_open = sum(1 for j, _ in self.nb[c] if self.pas[j])
        gap = abs(self.room[c] - self.room[g])
        ncand = len(self.cands0)
        return [self.dist_g[c] / 30.0, self.manh(c, g) / 30.0, gap / 5.0, self.room[c] / 100.0,
                span / 40.0, (span - ecc_g) / 10.0, edge / 5.0, nb_open / 4.0,
                (self.room[c] - self.room[g]) / 5.0, 1.0 / max(1, ncand)]

    # ------------------------------------------------------------ bookkeeping
    def update(self, obs, T, O, A):
        t = self.turn
        n = self.n
        my_struct = 0
        my_castles = set()
        vis_enemy_army = 0
        vis_enemy_cells = 0
        vis_enemy_castles = set()
        new_struct_fog = []
        for i in range(n):
            ti = T[i]
            if ti == 0 or ti == 5:
                if ti == 5 and not self.mountain[i] and not self.castle[i]:
                    new_struct_fog.append(i)
                continue
            self.ever_visible[i] = True
            self.last_seen[i] = t
            oi = O[i]
            self.last_owner[i] = oi
            self.last_army[i] = A[i]
            if ti == 3:
                self.castle[i] = True
            if oi == 1:
                if ti == 3 or ti == 4:
                    my_struct += 1
                    if ti == 3:
                        my_castles.add(i)
            elif oi == 2:
                vis_enemy_army += A[i]
                vis_enemy_cells += 1
                if self.first_enemy_seen[i] < 0:
                    self.first_enemy_seen[i] = t
                if ti == 4:
                    self.egen = i
                if ti == 3:
                    vis_enemy_castles.add(i)
            if self.cands and i in self.cands and not (ti == 4 and oi == 2):
                self.cands.discard(i)
        self.T, self.O, self.A = T, O, A
        self.my_castles = my_castles
        self.vis_enemy_army = vis_enemy_army
        self.vis_enemy_cells = vis_enemy_cells
        my_army = int(obs["my_army"])
        my_land = int(obs["my_land"])
        opp_army = int(obs["opp_army"])
        opp_land = int(obs["opp_land"])
        self.my_army, self.my_land, self.opp_army, self.opp_land = my_army, my_land, opp_army, opp_land

        # castles that changed hands (visible)
        for i in vis_enemy_castles:
            self.enemy_castles.add(i)
        for i in list(self.enemy_castles):
            if O[i] == 1:
                self.enemy_castles.discard(i)
        prev = self.prev
        if prev is not None and prev["turn"] == t - 1:
            lost = prev["my_castles"] - my_castles
            for i in lost:  # we lost a castle -> it is the enemy's now
                self.enemy_castles.add(i)
            gained_from_enemy = {i for i in my_castles - prev["my_castles"] if prev["enemy_castles"] and i in prev["enemy_castles"]}
            # our growth / build bookkeeping
            g2 = 1 if t % 2 == 0 else 0
            g50 = 1 if t % 50 == 0 else 0
            our_build = 0
            la = self.last_action
            if la[0] == 2:
                bi = la[1] * self.W + la[2]
                if bi in my_castles and bi not in prev["my_castles"]:
                    our_build = prev["build_cost_cache"].get(bi, self.price_for(prev["my_structs"], bi))
            X = prev["my_army"] - our_build + g2 * my_struct + g50 * my_land - my_army
            R = opp_army - prev["opp_army"] + X - g50 * opp_land
            S_prev = 1 + len(prev["enemy_castles"])
            S_now = S_prev + len(lost) - len(gained_from_enemy)
            built = 0
            if g2:
                if R != S_now:
                    built = S_now + 1 - R
            else:
                built = -R
            if built >= 35:
                cell = self.locate_new_castle(new_struct_fog, vis_enemy_castles, prev)
                if cell >= 0:
                    self.castle[cell] = True
                    self.register_enemy_build(cell, built, prev["enemy_castles"])
                    self.enemy_castles.add(cell)
            elif built != 0:
                self.acct_desync = t
        for i in new_struct_fog:  # any unexplained new structure is a castle too
            self.castle[i] = True
            self.enemy_castles.add(i)
        self.enemy_struct = 1 + len(self.enemy_castles)
        my_structs = [i for i in my_castles] + ([self.general] if self.general >= 0 else [])
        self.my_structs = my_structs
        self.prev = {"turn": t, "my_army": my_army, "opp_army": opp_army, "my_castles": set(my_castles),
                     "enemy_castles": set(self.enemy_castles), "my_structs": list(my_structs),
                     "build_cost_cache": {}}
        # belief pruning from land count / first sightings
        self.prune_candidates()
        self.update_tracks()

    def update_tracks(self):
        """Track big enemy stacks through fog (position, army, last seen turn)."""
        t = self.turn
        A, O, T = self.A, self.O, self.T
        big = max(PARAMS["track_min"], int(PARAMS["track_frac"] * self.opp_army))
        seen = [i for i in range(self.n) if O[i] == 2 and A[i] >= big]
        tracks = self.tracks
        used = set()
        for e in sorted(seen, key=lambda i: -A[i]):
            best, bd = None, None
            for k, tr in enumerate(tracks):
                if k in used:
                    continue
                d = self.manh(e, tr[0])
                if d <= (t - tr[2]) + 1 and (bd is None or d < bd):
                    bd, best = d, k
            if best is None:
                tracks.append([e, A[e], t])
                used.add(len(tracks) - 1)
            else:
                tracks[best] = [e, A[e], t]
                used.add(best)
        keep = []
        for k, tr in enumerate(tracks):
            if k in used:
                keep.append(tr)
                continue
            c = tr[0]
            if t - tr[2] > PARAMS["track_ttl"]:
                continue
            if T[c] != 0 and T[c] != 5 and O[c] == 1 and t - tr[2] <= 1:
                # we took its cell: it fought us; keep only if a big enemy cell is adjacent
                continue
            keep.append(tr)
        self.tracks = keep[:8]

    def locate_new_castle(self, new_fog, vis_enemy_castles, prev):
        for i in new_fog:
            return i
        for i in vis_enemy_castles:
            if i not in prev["enemy_castles"]:
                return i
        return -1

    def register_enemy_build(self, cell, price, prev_castles):
        self.enemy_builds.append((self.turn, cell, price))
        surcharge = price - 35
        W = self.W
        for c in prev_castles:
            d = self.manh(c, cell)
            if d <= 6:
                surcharge -= 14 - 2 * d
        if surcharge < 0 or surcharge > 12 or surcharge % 2:
            return
        if surcharge > 0:
            ring = (14 - surcharge) // 2
            self.rings.append((cell, ring))
        else:
            self.rings.append((cell, None))
        self.apply_rings()

    def apply_rings(self):
        if self.egen >= 0 or not self.cands:
            return
        keep = set()
        for c in self.cands:
            ok = True
            for cell, ring in self.rings:
                d = self.manh(c, cell)
                if ring is None:
                    if d <= 6:
                        ok = False
                        break
                elif d != ring:
                    ok = False
                    break
            if ok:
                keep.add(c)
        if keep:
            self.cands = keep

    def prune_candidates(self):
        if self.egen >= 0:
            self.cands = {self.egen}
            return
        if not self.cands:
            # belief collapsed (shouldn't happen): fall back to unseen far cells
            self.cands = {i for i in self.cands0 if not self.ever_visible[i]} or set(self.cands0)
            return
        # an enemy cell e seen at turn te must be within reach of the general:
        # dist(general, e) <= te  and (connected territory) dist <= opp_land
        enemy_seen = [i for i in range(self.n) if self.last_owner[i] == 2 and self.first_enemy_seen[i] >= 0]
        if not enemy_seen:
            return
        keep = set()
        for c in self.cands:
            dc = self.cand_dist.get(c)
            if dc is None:
                keep.add(c)
                continue
            ok = True
            for e in enemy_seen:
                if dc[e] > self.first_enemy_seen[e] + 1:
                    ok = False
                    break
            if ok:
                keep.add(c)
        if keep:
            self.cands = keep

    def price_for(self, structs, i):
        cost = 35
        W = self.W
        r, c = divmod(i, W)
        for j in structs:
            rj, cj = divmod(j, W)
            d = abs(rj - r) + abs(cj - c)
            if d <= 6:
                cost += 14 - 2 * d
        return cost

    # ------------------------------------------------------------ belief API
    def belief_scores(self, frm=None):
        """Lower is better: exploration cost plus distance to recent enemy land."""
        cands = self.cands or set(self.cands0)
        if not cands:
            return {}
        t = self.turn
        if frm is None:
            frm = [i for i in range(self.n) if self.O[i] == 1]
        elif isinstance(frm, int):
            frm = [frm]
        dfrm = self.bfs(frm) if frm else self.dist_g
        recent = [i for i in range(self.n) if self.last_owner[i] == 2 and self.last_seen[i] >= t - 80]
        de = self.bfs(recent) if recent else None
        P = PARAMS
        out = {}
        pr = self.prior
        for c in cands:
            sc = P["belief_explore_w"] * min(dfrm[c], 60)
            if de is not None:
                sc += P["belief_enemy_w"] * min(de[c], 40)
            sc -= P["belief_prior_w"] * pr.get(c, 0.0)
            out[c] = sc
        return out

    def belief_target(self, frm=None):
        """Most likely / cheapest-to-check enemy general cell."""
        if self.egen >= 0:
            return self.egen
        sc = self.belief_scores(frm)
        if not sc:
            return -1
        return min(sc, key=sc.get)

    # ---------------------------------------------------------- local resolve
    def resolve(self, mine_act, en_act):
        """Exact one-turn resolution on the visible board. Returns (my_gen_lost, en_gen_taken)."""
        A, O = self.A, self.O
        ov = {}

        def get(i):
            if i in ov:
                return ov[i]
            return (O[i], A[i])

        def amount(owner, a):
            k = a[0]
            if k != 0:
                return None
            r, c, d, sp = a[1], a[2], a[3], a[4]
            src = r * self.W + c
            dr, dc = DIRS[d]
            rr, cc = r + dr, c + dc
            if not (0 <= rr < self.H and 0 <= cc < self.W):
                return None
            dst = rr * self.W + cc
            o, arm = get(src)
            if o != owner or self.mountain[dst]:
                return None
            amt = arm // 2 if sp == 1 else arm - 1
            amt = min(amt, arm - 1)
            if amt <= 0:
                return None
            return src, dst, amt

        # builds first
        if mine_act[0] == 2:
            bi = mine_act[1] * self.W + mine_act[2]
            o, arm = get(bi)
            ov[bi] = (o, arm - self.price_for(self.my_structs, bi))
            mine_act = PASS
        if en_act[0] == 2:
            en_act = PASS
        acts = {1: mine_act, 2: en_act}

        def key(owner):
            a = acts[owner]
            other = acts[3 - owner]
            if a[0] != 0:
                return (True, True, 2 ** 31, 0)
            dr, dc = DIRS[a[3]]
            di, dj = a[1] + dr, a[2] + dc
            chasing = other[0] == 0 and di == other[1] and dj == other[2]
            ci = min(max(di, 0), self.H - 1) * self.W + min(max(dj, 0), self.W - 1)
            reinf = get(ci)[0] == owner
            arm = get(min(max(a[1], 0), self.H - 1) * self.W + min(max(a[2], 0), self.W - 1))[1]
            pidx = self.pid if owner == 1 else 1 - self.pid
            return (not chasing, not reinf, arm, pidx)

        order = (1, 2) if key(1) < key(2) else (2, 1)
        late = self.turn >= 800
        egen = self.egen
        result = {1: False, 2: False}  # owner -> captured/touched the other general
        for owner in order:
            m = amount(owner, acts[owner])
            if m is None:
                continue
            src, dst, amt = m
            target_gen = self.general if owner == 2 else egen
            if late and dst == target_gen and target_gen >= 0:
                result[owner] = True
            o, arm = get(dst)
            if o == owner:
                ov[dst] = (o, arm + amt)
            else:
                if amt > arm:
                    ov[dst] = (owner, amt - arm)
                    if dst == target_gen and target_gen >= 0:
                        result[owner] = True
                else:
                    ov[dst] = (o, arm - amt)
            so, sa = get(src)
            ov[src] = (so, sa - amt)
        return result[2], result[1]

    def enemy_general_threats(self):
        """Enemy candidate moves that could hit our general this turn."""
        g = self.general
        out = []
        for j, _ in self.nb[g]:
            if self.O[j] == 2 and self.A[j] >= 2:
                for sp in (0, 1):
                    a = self.mv(j, g, sp)
                    if a:
                        out.append(a)
        return out

    def safe(self, act, threats=None):
        if threats is None:
            threats = self.enemy_general_threats()
        for ea in threats:
            lost, won = self.resolve(act, ea)
            if lost and not won:
                return False
        return True

    # ---------------------------------------------------------------- gather
    def gather_tree(self, root, allowed=None):
        O, A = self.O, self.A
        parent = {root: -1}
        depth = {root: 0}
        order = [root]
        dq = deque([root])
        while dq:
            i = dq.popleft()
            for j, _ in self.nb[i]:
                if j not in parent and O[j] == 1 and (allowed is None or allowed(j)):
                    parent[j] = i
                    depth[j] = depth[i] + 1
                    order.append(j)
                    dq.append(j)
        return parent, depth, order

    def gather_select(self, root, need=None, budget=None, allowed=None):
        """Greedy connected-subtree selection for gathering into root.

        Repeatedly adds the root-ward path with the best army-per-move ratio
        until the move budget is spent or `need` army is collected.
        Returns (parent, depth, selected set incl. root, collected value)."""
        A = self.A
        parent, depth, order = self.gather_tree(root, allowed)
        if budget is None:
            budget = PARAMS["gather_default_budget"]
        val = {i: A[i] - 1 for i in order}
        val[root] = 0
        sel = {root}
        total = 0
        used = 0
        rest = order[1:]
        while used < budget and (need is None or total < need):
            if self.late(0.8):
                break
            gain, cost = {root: 0}, {root: 0}
            best, br, bc = -1, 0.0, 0
            room = budget - used
            for u in rest:
                p = parent[u]
                if u in sel:
                    gain[u], cost[u] = 0, 0
                    continue
                gu = val[u] + gain[p]
                cu = 1 + cost[p]
                gain[u], cost[u] = gu, cu
                if cu <= room and gu > 0:
                    r = gu / cu
                    if need is not None and gu >= need - total:
                        r += 1000.0 / cu  # finishing the job cheaply beats ratio
                    if r > br:
                        best, br, bc = u, r, cu
            if best < 0:
                break
            u = best
            while u not in sel:
                sel.add(u)
                total += val[u]
                used += 1
                u = parent[u]
        return parent, depth, sel, total

    def gather_move(self, root, need=None, budget=None, allowed=None):
        A = self.A
        parent, depth, sel, total = self.gather_select(root, need, budget, allowed)
        best, bk = -1, None
        has_child_army = set()
        for i in sel:
            if i != root and A[i] >= 2:
                has_child_army.add(parent[i])
        for i in sel:
            if i == root or A[i] < 2:
                continue
            if i in has_child_army:
                continue
            k = (depth[i], A[i])
            if bk is None or k > bk:
                bk, best = k, i
        if best < 0:
            return None, total
        return self.mv(best, parent[best]), total

    # -------------------------------------------------------------- pathing
    def path_to(self, src, dst, enemy_cost=1.0, avoid_general=True):
        """Cheapest path src->dst by Dijkstra (cost: 1 per step + army to beat).
        Never routes through our own general (a stack merging into it would
        then march the garrison out)."""
        A, O = self.A, self.O
        gen = self.general if (avoid_general and src != self.general) else -1
        n = self.n
        dist = [INF] * n
        prev = [-1] * n
        dist[src] = 0
        h = [(0, src)]
        while h:
            d, i = heappop(h)
            if d > dist[i]:
                continue
            if i == dst:
                break
            for j, _ in self.nb[i]:
                if not self.pas[j] or j == gen:
                    continue
                if O[j] == 1:
                    c = 1
                elif O[j] == 2 or (self.T[j] == 3):
                    c = 1 + enemy_cost * A[j]
                elif self.T[j] == 5 or (self.castle[j] and self.T[j] == 0):
                    c = 1 + enemy_cost * max(20, self.last_army[j])
                else:
                    c = 2 if self.T[j] != 0 else 2
                nd = d + c
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heappush(h, (nd, j))
        if dist[dst] >= INF:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # ----------------------------------------------------------------- decide
    def decide(self):
        t = self.turn
        T, O, A = self.T, self.O, self.A
        g = self.general
        threats = self.enemy_general_threats()

        a = self.win_now()
        if a:
            return a
        a = self.urgent_defense(threats)
        if a:
            return a
        if t < PARAMS["open_end"]:
            a = self.opening()
            if a and self.safe(a, threats):
                return a
        a = self.macro()
        if a is None:
            a = PASS
        if a[0] != 1 and not self.safe(a, threats):
            alt = self.defensive_alternatives(threats)
            if alt:
                return alt
        return a

    # ---------------------------------------------------------------- winning
    def win_now(self):
        eg = self.egen
        if eg < 0:
            return None
        A, O = self.A, self.O
        t = self.turn
        best = None
        for j, _ in self.nb[eg]:
            if O[j] == 1 and A[j] >= 2:
                if t >= 800 or A[j] - 1 > A[eg]:
                    if best is None or A[j] > A[best]:
                        best = j
        if best is not None:
            return self.mv(best, eg)
        return None

    def urgent_defense(self, threats):
        if not threats:
            return None
        if self.safe(PASS, threats):
            return None
        # find a move that makes us safe; prefer chase kills of the attacker source,
        # then reinforcing the general with the largest neighbor stack.
        g = self.general
        best, bk = None, None
        for a in self.defensive_candidates(threats):
            if self.safe(a, threats):
                arm = self.A[a[1] * self.W + a[2]]
                k = (1 if a[0] == 0 else 0, arm)
                if bk is None or k > bk:
                    bk, best = k, a
        if best:
            return best
        # cannot be fully safe: if we can win/draw by touching first, do it
        return None

    def defensive_candidates(self, threats):
        g = self.general
        A, O = self.A, self.O
        out = []
        srcs = {t[1] * self.W + t[2] for t in threats}
        # chase the attacker source from a third tile
        for s in srcs:
            for z, _ in self.nb[s]:
                if z != g and O[z] == 1 and A[z] >= 2:
                    out.append(self.mv(z, s))
            if A[g] >= 2:
                out.append(self.mv(g, s))
        # reinforce general
        for j, _ in self.nb[g]:
            if O[j] == 1 and A[j] >= 2:
                out.append(self.mv(j, g))
        return [a for a in out if a]

    def defensive_alternatives(self, threats):
        best, bk = None, None
        for a in self.defensive_candidates(threats) + [PASS]:
            if self.safe(a, threats):
                arm = self.A[a[1] * self.W + a[2]] if a[0] == 0 else 0
                if bk is None or arm > bk:
                    bk, best = arm, a
        return best

    # ---------------------------------------------------------------- opening
    def opening(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        g = self.general
        cfg = self.open_cfg
        # 1) continue a wave: a non-general own cell with >= 2 adjacent to neutral
        best, bk = None, None
        for i in range(self.n):
            if O[i] == 1 and i != g and A[i] >= 2:
                for j, _ in self.nb[i]:
                    if self.pas[j] and O[j] == 0 and not self.castle[j]:
                        k = (self.open_score(j), A[i])
                        if bk is None or k > bk:
                            bk, best = k, (i, j)
        if best:
            return self.mv(*best)
        # 2) stack stuck inside own land: move it toward nearest neutral
        stuck = [i for i in range(self.n) if O[i] == 1 and i != g and A[i] >= 2]
        if stuck:
            i = max(stuck, key=lambda k: A[k])
            step = self.step_to_frontier(i)
            if step is not None:
                return self.mv(i, step)
        # 3) launch from general
        a = A[g]
        if a >= 2:
            dfront, step = self.frontier_info(g)
            if step is not None:
                transit = max(0, dfront - 1)
                need = int((49 - t - transit) / cfg["div"])
                if a - 1 >= need or (t >= cfg["late"] and a >= 2):
                    return self.mv(g, step)
        return PASS

    def open_score(self, j):
        """Prefer open areas, away from the general, toward the enemy candidates."""
        cfg = self.open_cfg
        free = 0
        for k, _ in self.nb[j]:
            if self.pas[k] and self.O[k] == 0:
                free += 1
        dc = self.open_dc
        toward = -dc[j] * cfg["w_toward"] if dc else 0.0
        return cfg["w_free"] * free + cfg["w_dist"] * min(self.dist_g[j], 12) + toward

    def plan_opening(self):
        """Pick the opening configuration that maximises land at turn 50 in a
        single-player simulation (the opponent cannot interfere this early)."""
        g = self.general
        tgt = max(self.prior, key=self.prior.get) if self.prior else -1
        self.open_dc = self.cand_dist.get(tgt) if tgt >= 0 else None
        best, bkey = None, None
        save = (getattr(self, "O", None), getattr(self, "A", None), getattr(self, "T", None), self.turn)
        deadline = time.perf_counter() + PARAMS["open_plan_s"]
        for div in (2.0, 1.7, 2.4, 1.5):
            for w_free, w_dist in ((1.0, 0.3), (1.0, 0.0), (0.5, 0.6), (1.5, 0.3)):
                for w_toward in (0.15, 0.0):
                    if time.perf_counter() > deadline:
                        break
                    cfg = {"div": div, "w_free": w_free, "w_dist": w_dist, "w_toward": w_toward, "late": 44}
                    land50, frontier = self.sim_opening(cfg)
                    key = (land50, frontier + (1 if w_toward > 0 else 0))
                    if bkey is None or key > bkey:
                        bkey, best = key, cfg
        self.O, self.A, self.T, self.turn = save
        self.open_cfg = best or {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}
        self.open_expect = bkey

    def sim_opening(self, cfg):
        n = self.n
        g = self.general
        O = [0] * n
        A = [0] * n
        O[g] = 1
        A[g] = 1
        self.T = [1] * n
        self.open_cfg = cfg
        W = self.W
        for t in range(50):
            self.O, self.A, self.turn = O, A, t
            a = self.opening()
            if a and a[0] == 0:
                i = a[1] * W + a[2]
                dr, dc = DIRS[a[3]]
                j = (a[1] + dr) * W + a[2] + dc
                amt = A[i] // 2 if a[4] == 1 else A[i] - 1
                if amt > 0 and O[i] == 1 and self.pas[j]:
                    A[i] -= amt
                    if O[j] == 1:
                        A[j] += amt
                    else:
                        O[j] = 1
                        A[j] = amt
            if (t + 1) % 2 == 0:
                A[g] += 1
        land = sum(O)
        frontier = 0
        for i in range(n):
            if O[i] == 0 and self.pas[i]:
                for j, _ in self.nb[i]:
                    if O[j] == 1:
                        frontier += 1
                        break
        return land, frontier

    def frontier_info(self, src):
        """(distance to nearest neutral cell through own cells, first step)."""
        O = self.O
        prev = {src: -1}
        dq = deque([src])
        while dq:
            i = dq.popleft()
            for j, _ in self.nb[i]:
                if j in prev or not self.pas[j]:
                    continue
                if O[j] == 0 and not self.castle[j]:
                    prev[j] = i
                    # reconstruct first step
                    path = [j]
                    while prev[path[-1]] != src:
                        path.append(prev[path[-1]])
                    return len(path), path[-1]
                if O[j] == 1:
                    prev[j] = i
                    dq.append(j)
        return INF, None

    def step_to_frontier(self, i):
        d, step = self.frontier_info(i)
        return step

    # ------------------------------------------------------------------ macro
    def macro(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        g = self.general
        P = PARAMS

        if t >= P["fortress_turn"]:
            a = self.endgame()
            if a:
                return a
        a = self.try_kill()
        if a:
            return a

        need_g = self.garrison_need()
        self.need_g = need_g
        self.enemy_dist = self.enemy_distance_map()

        options = []  # (score, action)
        cap = self.best_capture(need_g)
        if cap:
            options.append(cap + ("capture",))
        if A[g] < need_g:
            eta = self.threat_eta
            budget = None if eta >= INF else max(1, eta - 1)
            a, tot = self.gather_move(g, need=need_g - A[g], budget=budget)
            if a:
                urgent = eta <= P["track_threat_dist"]
                options.append((P["w_garrison_urgent"] if urgent else P["w_garrison"], a, "garrison"))
        b = self.castle_build_now()
        if b:
            options.append(b + ("build",))
        sc = self.scout_move() if not self.late(0.5) else None
        if sc:
            options.append(sc + ("scout",))
        hf = self.home_fill_move(need_g) if not self.late(0.5) else None
        if hf:
            options.append(hf + ("home",))
        self.pending_stack = None
        c = self.cycle_move(need_g) if not self.late(0.6) else None
        if c:
            launching = self.cyc is not None and self.cyc.get("mode") == "launch"
            options.append((P["w_launch"] if launching else P["w_cycle"], c,
                            ("launch" if launching else "cyc_" + str(self.cyc.get("purpose") if self.cyc else ""))))
        if not options:
            a, tot = self.gather_move(g, budget=10)
            self.last_label = "idle_gather" if a else "pass"
            return a or PASS
        options.sort(key=lambda x: -x[0])
        choice = options[0][1]
        self.last_label = options[0][2]
        if c is not None and choice is c and self.cyc is not None:
            if self.cyc.get("mode") == "launch" and self.pending_stack is not None:
                self.cyc["stack"] = self.pending_stack
            elif self.cyc.get("mode") == "gather":
                self.cyc["moves"] = self.cyc.get("moves", 0) + 1
        return choice

    # -------------------------------------------------------------- garrison
    def garrison_need(self):
        """Army the general should hold, and the time (eta) we have to get it.

        The binding threat (largest requirement) sets self.threat_eta, which is
        the move budget for defensive gathering."""
        A, O = self.A, self.O
        g = self.general
        t = self.turn
        P = PARAMS
        need = P["garrison_min"]
        eta = INF
        for i in range(self.n):
            if O[i] == 2 and A[i] >= 3:
                d = self.dist_g[i]
                if d < P["threat_vis_range"]:
                    ni = A[i] - int((d - 1) * P["threat_decay"]) + P["threat_margin"]
                    if ni > need:
                        need, eta = ni, d
        for c, army, ts in self.tracks:
            d = max(1, self.dist_g[c] - (t - ts))
            if d <= P["track_threat_dist"]:
                ni = army + P["threat_margin"]
                if ni > need:
                    need, eta = ni, d
        hidden = self.opp_army - self.vis_enemy_army - max(0, self.opp_land - self.vis_enemy_cells - 1)
        if hidden > 0:
            wd = self.fog_distance()
            hidden = min(hidden, int(self.opp_army * P["hidden_stack_frac"]))
            if wd <= 4:
                f = P["garrison_frac_hidden"]
            elif wd <= 7:
                f = P["garrison_frac_hidden"] * 0.6
            else:
                f = P["garrison_frac_hidden"] * 0.3
            ni = int(hidden * f)
            if ni > need:
                need, eta = ni, max(2, wd)
        self.threat_eta = eta
        cap = int(self.my_army * P["garrison_cap_frac"])
        return min(need, max(P["garrison_min"], cap))

    def fog_distance(self):
        T = self.T
        best = INF
        dg = self.dist_g
        for i in range(self.n):
            if T[i] == 0 and dg[i] < best:
                best = dg[i]
        return best

    def enemy_distance_map(self):
        cells = [i for i in range(self.n) if self.last_owner[i] == 2 and self.last_seen[i] >= self.turn - 60]
        cells += list(self.enemy_castles)
        if not cells:
            return [INF] * self.n
        return self.bfs(cells)

    # -------------------------------------------------------------- captures
    def best_capture(self, need_g):
        """Best single capture move with a heuristic value (score, action)."""
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        g = self.general
        t = self.turn
        small_cap = max(P["small_min"], int(P["small_frac"] * self.my_army))
        bonus = t % 50 >= 50 - P["bonus_lead"] or t % 50 < P["bonus_window"]
        tgt = self.belief_target()
        dc = self.cand_dist.get(tgt) if tgt >= 0 else None
        best, bs = None, None
        for i in range(self.n):
            ai = A[i]
            if O[i] != 1 or ai < 2:
                continue
            is_g = i == g
            for j, d in self.nb[i]:
                if not self.pas[j] or O[j] == 1:
                    continue
                aj = A[j]
                split = 0
                send = ai - 1
                if is_g:
                    if ai - 1 - aj >= 1 and ai - (ai - 1) >= need_g:
                        split = 0
                    elif ai // 2 > aj and ai - ai // 2 >= need_g:
                        split, send = 1, ai // 2
                    else:
                        continue
                if send <= aj:
                    continue
                if O[j] == 2:
                    v = P["v_enemy"] + P["v_kill"] * aj
                    if T[j] == 3:
                        v += P["v_ecastle"]
                    if self.manh(j, g) <= 3:
                        v += P["v_near_home"]
                else:
                    if T[j] == 3:
                        continue
                    dgj = self.dist_g[j]
                    home = max(0, P["home_r"] - dgj) / P["home_r"]
                    if ai > small_cap and not is_g and not (home > 0 and ai <= 3 * small_cap):
                        continue
                    v = P["v_neutral"] * (P["bonus_mult"] if bonus else 1.0) + P["v_home"] * home
                    if dc:
                        v -= P["toward_w"] * min(dc[j], 30) / 30.0
                    v -= 0.002 * ai
                if is_g:
                    v -= P["gen_move_pen"]
                if bs is None or v > bs:
                    bs, best = v, [0, i // self.W, i % self.W, d, split]
        if best is None:
            return None
        return (bs, best)

    # ------------------------------------------------------------- home zone
    def home_fill_move(self, need_g):
        """Own every cell within home_r of the general: vision = warning time."""
        t = self.turn
        P = PARAMS
        if t < P["home_fill_start"]:
            return None
        A, O = self.A, self.O
        g = self.general
        R = P["home_r"]
        dg = self.dist_g
        holes = [j for j in range(self.n) if self.pas[j] and O[j] != 1 and dg[j] <= R
                 and not (self.castle[j] and O[j] == 0)]
        if not holes:
            return None
        best, bk = None, None
        for j in holes:
            for i, d in self.nb[j]:
                if O[i] != 1 or A[i] < 2:
                    continue
                if i == g:
                    send = A[i] // 2
                    if send <= A[j] or A[i] - send < need_g:
                        continue
                    split = 1
                else:
                    send = A[i] - 1
                    if send <= A[j]:
                        continue
                    split = 0
                k = (-dg[j], -A[i] if i != g else -10 ** 6)
                if bk is None or k > bk:
                    dd = (d ^ 1)  # direction from i to j is the reverse of j->i
                    bk, best = k, [0, i // self.W, i % self.W, dd, split]
        if best is None:
            # bring a small stack next to the nearest hole
            return None
        return (P["w_home_fill"], best)

    # ---------------------------------------------------------------- scouting
    def scout_move(self):
        """Walk a modest stack toward the nearest unexplored general candidate."""
        t = self.turn
        P = PARAMS
        if self.egen >= 0 or t < P["scout_start"] or not self.cands:
            return None
        A, O = self.A, self.O
        g = self.general
        cyc = self.cyc
        busy = set()
        if cyc:
            busy = {cyc.get("root"), cyc.get("stack")}
        lo, hi = P["scout_min"], max(P["scout_min"] + 1, int(P["scout_max_frac"] * self.my_army))
        srcs = [i for i in range(self.n) if O[i] == 1 and lo <= A[i] <= hi and i != g and i not in busy]
        if not srcs:
            return None
        dc = self.bfs(list(self.cands))
        i = min(srcs, key=lambda k: (dc[k], -A[k]))
        if dc[i] >= INF or dc[i] == 0:
            return None
        best, bv = None, None
        for j, d in self.nb[i]:
            if not self.pas[j] or dc[j] >= dc[i]:
                continue
            if O[j] != 1 and A[i] - 1 <= A[j]:
                continue
            v = -A[j] if O[j] != 1 else 0
            if bv is None or v > bv:
                bv, best = v, j
        if best is None:
            return None
        return (P["w_scout"], self.mv(i, best))

    # --------------------------------------------------------------- castles
    def castle_build_now(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        if not (P["castle_start"] <= t <= P["castle_stop"]):
            return None
        horizon = P["castle_horizon"]
        de = self.enemy_dist
        structs = self.my_structs
        best, bv = None, None
        cyc = self.cyc
        reserved = set()
        if cyc and cyc.get("purpose") == "attack":
            reserved = {cyc.get("root"), cyc.get("stack")}
        for i in range(self.n):
            if O[i] == 1 and T[i] == 1 and A[i] >= 35:
                if i in reserved:
                    continue
                if de[i] < P["castle_safe_dist"]:
                    continue
                price = self.price_for(structs, i)
                if A[i] < price + P["castle_reserve"]:
                    continue
                v = 0.5 * (horizon - t) - price * P["castle_price_w"]
                if bv is None or v > bv:
                    bv, best = v, i
        if best is None or bv < P["castle_val_min"]:
            return None
        if self.threat_eta <= P["track_threat_dist"] and self.A[self.general] < self.need_g:
            return None
        return (P["w_build"], [2, best // self.W, best % self.W, 0, 0])

    def n_castles_wanted(self):
        t = self.turn
        P = PARAMS
        if t < P["castle_start"]:
            return 0
        return 1 + int((t - P["castle_start"]) / P["castle_every"])

    def castle_site(self):
        """Best site to gather a castle at: (cell, price) or None."""
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        de = self.enemy_dist
        structs = self.my_structs
        g = self.general
        best, bv = None, None
        for i in range(self.n):
            if O[i] != 1 or T[i] != 1 or i == g:
                continue
            if de[i] < P["castle_safe_dist"] + 1:
                continue
            price = self.price_for(structs, i)
            dg = self.dist_g[i]
            v = -price * P["castle_price_w"] - P["castle_move_w"] * max(0, dg - 7) * 0.5
            v += min(de[i], 12) * P["castle_safety_w"]
            near_home = sum(1 for c in self.my_castles if self.dist_g[c] <= P["castle_home_maxd"])
            if near_home < P["castle_home_n"]:
                v -= P["castle_home_w"] * max(0, dg - P["castle_home_maxd"])
            if bv is None or v > bv:
                bv, best = v, (i, price)
        return best

    # ------------------------------------------------------------ army cycle
    def choose_target(self, frm=None):
        """Target for the attack stack."""
        A, O, T = self.A, self.O, self.T
        g = self.general
        P = PARAMS
        best, bk = -1, None
        home = [g] + list(self.my_castles)
        for i in range(self.n):
            if O[i] == 2 and A[i] >= 4:
                d = min(self.manh(i, h) for h in home)
                if d <= P["intercept_dist"]:
                    k = A[i] - 3 * d
                    if bk is None or k > bk:
                        bk, best = k, i
        if best >= 0:
            return best
        if self.egen >= 0:
            return self.egen
        if self.enemy_castles and frm is not None:
            df = self.bfs([frm])
            c = min(self.enemy_castles, key=lambda c: df[c])
            est = self.last_army[c] + (self.turn - max(0, self.last_seen[c])) // 2
            if df[c] < 12 and A[frm] > est + df[c] + 5:
                return c
        return self.belief_target(frm)

    def cycle_move(self, need_g):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        g = self.general
        cyc = self.cyc
        if cyc is None or cyc.get("turn") is None:
            cyc = self.cyc = self.new_cycle(need_g)
            if cyc is None:
                return None
        if cyc["mode"] == "gather":
            root = cyc["root"]
            if O[root] != 1:
                self.cyc = None
                return None
            if cyc["purpose"] == "castle":
                site, price = cyc["site"], cyc["price"]
                if A[root] >= price + P["castle_reserve"]:
                    self.cyc = None  # build handled by castle_build_now next turn
                    return [2, root // self.W, root % self.W, 0, 0]
                need = price + P["castle_reserve"] - A[root]
            else:
                need = None
            elapsed = cyc.get("moves", 0)
            if t - cyc["turn"] > 3 * cyc["budget"] + 5:
                elapsed = cyc["budget"]
            if elapsed < cyc["budget"]:
                # feed the general's surplus toward the root first
                surplus = A[g] - need_g
                if surplus >= P["feed_min"] and root != g:
                    path = self.bfs_path_own(g, root)
                    if path and len(path) >= 2:
                        split = 0 if need_g <= 1 or A[g] - 1 < need_g else 1
                        if split == 0 and A[g] - (A[g] - 1) < need_g:
                            split = 1
                        if split == 1 and A[g] - A[g] // 2 < need_g:
                            split = None
                        if split is not None:
                            return self.mv(g, path[1], split)
                a, tot = self.gather_move(root, need=need, budget=cyc["budget"] - elapsed,
                                          allowed=lambda j: j != g)
                if a:
                    return a
            if cyc["purpose"] == "castle":
                # could not gather enough: give up this castle
                self.cyc = None
                return None
            if root == g:
                # launch half (or all but the garrison) of the general's stack
                tgt = self.choose_target(g)
                send_half = A[g] // 2
                send_all = A[g] - 1
                if A[g] - 1 - send_all >= need_g - 1 and send_all >= P["min_stack"]:
                    split, send = 0, send_all
                elif A[g] - send_half >= need_g and send_half >= P["min_stack"]:
                    split, send = 1, send_half
                else:
                    self.cyc = None
                    return None
                path = self.path_to(g, tgt) if tgt >= 0 else None
                if not path or len(path) < 2:
                    self.cyc = None
                    return None
                nxt = path[1]
                if O[nxt] != 1 and send <= A[nxt]:
                    self.cyc = None
                    return None
                cyc["mode"] = "launch"
                cyc["stack"] = g
                self.pending_stack = nxt
                return self.mv(g, nxt, split)
            if A[root] >= P["min_stack"]:
                cyc["mode"] = "launch"
                cyc["stack"] = root
            else:
                self.cyc = None
                return None
        # launch
        s = cyc["stack"]
        if O[s] != 1 or A[s] < P["min_stack"] or s == g:
            self.cyc = None
            return None
        tgt = self.choose_target(s)
        if tgt < 0:
            self.cyc = None
            return None
        if s == tgt:
            self.cyc = None
            return None
        path = self.path_to(s, tgt)
        if not path or len(path) < 2:
            self.cyc = None
            return None
        nxt = path[1]
        if O[nxt] != 1 and A[s] - 1 <= A[nxt]:
            # blocked: re-gather into the stack where it stands
            self.cyc = {"mode": "gather", "purpose": "attack", "root": s, "turn": t,
                        "budget": P["regather_budget"], "moves": 0}
            return None
        self.pending_stack = nxt
        if t - cyc["turn"] > cyc["budget"] + P["launch_max"]:
            self.cyc = None
        return self.mv(s, nxt)

    def new_cycle(self, need_g):
        t = self.turn
        P = PARAMS
        g = self.general
        A, O = self.A, self.O
        if (P["castle_start"] <= t <= P["castle_stop"] and len(self.my_castles) < self.n_castles_wanted()
                and (self.last_purpose != "castle" or self.my_army < P["attack_min_army"])):
            site = self.castle_site()
            if site:
                cell, price = site
                self.last_purpose = "castle"
                return {"mode": "gather", "purpose": "castle", "root": cell, "site": cell,
                        "price": price, "turn": t, "budget": P["castle_gather_budget"]}
        tgt = self.choose_target()
        if tgt < 0:
            return None
        self.last_purpose = "attack"
        de = self.bfs([tgt])
        own = [i for i in range(self.n) if O[i] == 1 and i != g]
        root = g
        if own and P["attack_root_front"]:
            root = min(own, key=lambda i: (de[i], -A[i]))
        return {"mode": "gather", "purpose": "attack", "root": root, "turn": t,
                "budget": P["gather_budget"], "target": tgt, "moves": 0}

    def bfs_path_own(self, src, dst):
        O = self.O
        prev = {src: -1}
        dq = deque([src])
        while dq:
            i = dq.popleft()
            if i == dst:
                break
            for j, _ in self.nb[i]:
                if j not in prev and O[j] == 1:
                    prev[j] = i
                    dq.append(j)
        if dst not in prev:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # ---------------------------------------------------------------- attack
    def try_kill(self):
        eg = self.egen
        if eg < 0:
            return None
        A, O = self.A, self.O
        t = self.turn
        gen_army = A[eg] if self.T[eg] == 4 else self.last_army[eg] + (t - self.last_seen[eg]) // 2
        stacks = [i for i in range(self.n) if O[i] == 1 and A[i] >= 2]
        if not stacks:
            return None
        best = None
        for i in sorted(stacks, key=lambda k: -A[k])[:4]:
            if self.late(0.5):
                break
            is_g = i == self.general
            send = A[i] // 2 if is_g else A[i] - 1
            if is_g and A[i] - send < getattr(self, "need_g", 2):
                continue
            path = self.path_to(i, eg)
            if not path:
                continue
            cost = sum(A[j] for j in path[1:-1] if O[j] != 1) + len(path)
            need = gen_army + cost + PARAMS["kill_margin"] + (len(path) // 2)
            if t >= 800:
                need = cost + 2
            if send >= need:
                if best is None or len(path) < best[1]:
                    best = (i, len(path), path, 1 if is_g else 0)
        if best:
            return self.mv(best[2][0], best[2][1], best[3])
        return None

    # --------------------------------------------------------------- endgame
    def endgame(self):
        t = self.turn
        A, O = self.A, self.O
        g = self.general
        # clear enemy cells near our general
        best, bk = None, None
        for i in range(self.n):
            if O[i] == 2 and self.manh(i, g) <= 2:
                for j, _ in self.nb[i]:
                    if O[j] == 1 and A[j] - 1 > A[i] and j != g:
                        k = (-self.manh(i, g), A[i])
                        if bk is None or k > bk:
                            bk, best = k, (j, i)
        if best:
            return self.mv(*best)
        # own all orthogonal neighbours of the general
        for j, _ in self.nb[g]:
            if self.pas[j] and O[j] != 1:
                if A[g] - 1 > A[j]:
                    return self.mv(g, j, 0 if A[g] < 6 else 1)
        # deathtouch offence: stack adjacent to the enemy general at >= 800
        eg = self.egen
        if eg >= 0 and t >= PARAMS["dt_stage_turn"]:
            own = [i for i in range(self.n) if O[i] == 1 and A[i] >= 3]
            if own:
                de = self.bfs([eg])
                i = min(own, key=lambda k: (de[k], -A[k]))
                if de[i] > 1:
                    path = self.path_to(i, eg)
                    if path and len(path) >= 2:
                        nxt = path[1]
                        if nxt != eg and (O[nxt] == 1 or A[i] - 1 > A[nxt]):
                            return self.mv(i, nxt)
        return None


_BOT = None
_LAST_TURN = None


def _decide(obs):
    global _BOT, _LAST_TURN
    t = int(obs["turn"])
    H, W = int(obs["height"]), int(obs["width"])
    if (_BOT is None or _LAST_TURN is None or t <= _LAST_TURN or t == 0
            or _BOT.H != H or _BOT.W != W):
        _BOT = Bot(obs)
    _LAST_TURN = t
    b = _BOT
    b.t_start = time.perf_counter()
    b.budget_s = (PARAMS["first_budget_ms"] if t == 0 else PARAMS["soft_budget_ms"]) / 1000.0
    T, O, A = b.parse(obs)
    b.turn = t
    if b.general < 0:
        for i in range(b.n):
            if T[i] == 4 and O[i] == 1:
                b.general = i
                break
        if b.general < 0:
            return PASS
        b.setup(T)
        gc.collect()
        gc.freeze()
    b.update(obs, T, O, A)
    a = b.decide()
    return a


def _sanitize(a):
    try:
        k = int(a[0])
        if k not in (0, 1, 2):
            return [1, 0, 0, 0, 0]
        r, c, d, s = int(a[1]), int(a[2]), int(a[3]), int(a[4])
        if k == 1:
            return [1, 0, 0, 0, 0]
        if not (0 <= d <= 3):
            d = 0
        if s not in (0, 1):
            s = 0
        return [k, r, c, d, s]
    except Exception:
        return [1, 0, 0, 0, 0]


def act(observation):
    try:
        a = _sanitize(_decide(observation))
        if _BOT is not None:
            _BOT.last_action = a
        return a
    except Exception:
        if DEBUG:
            raise
        try:
            if _BOT is not None:
                _BOT.err += 1
                _BOT.last_action = [1, 0, 0, 0, 0]
        except Exception:
            pass
        return [1, 0, 0, 0, 0]

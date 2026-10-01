"""Pure-Python (stdlib-only) exact replica of the pinned competition engine.

Mirrors strakam/generals-bots@13db8f69 with --mode competition:
  generals/core/game.py            step, _determine_move_order, _execute_move,
                                   _apply_move, eliminate_player, global_update,
                                   _observe
  generals/modifiers/build_castles.py   apply_build_actions, build_cost_grid
  generals/modifiers/deathtouch.py      step (touch + mutual-capture draw)
  competition/protocol.py          encode_observation (type/owner grids)

Two players only. Board is stored as flat lists indexed i = r * W + c.
Kept dependency-free so the same code can be pasted into the submission for
tactical lookahead.
"""

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # UP, DOWN, LEFT, RIGHT
BASE_COST = 35
PROX_PEN = 14
PROX_DECAY = 2
PROX_RADIUS = (PROX_PEN - 1) // PROX_DECAY  # 6
DEATHTOUCH_TURN = 800
TRUNCATION = 1200

# observation type codes (competition/protocol.py)
T_FOG, T_PLAIN, T_MOUNTAIN, T_CASTLE, T_GENERAL, T_STRUCT_FOG = 0, 1, 2, 3, 4, 5


class State:
    """Mutable game state. owner[i]: -1 neutral, 0/1 player."""

    __slots__ = ("H", "W", "army", "owner", "mountain", "castle", "general",
                 "gpos", "eliminated", "time", "winner", "done")

    def __init__(self, H, W, mountains, gpos):
        self.H, self.W = H, W
        n = H * W
        self.army = [0] * n
        self.owner = [-1] * n
        self.mountain = [False] * n
        self.castle = [False] * n
        self.general = [False] * n
        for i in mountains:
            self.mountain[i] = True
        self.gpos = list(gpos)  # starting general cell per player (flat index)
        for p, g in enumerate(gpos):
            self.owner[g] = p
            self.general[g] = True
            self.army[g] = 1
        self.eliminated = [False, False]
        self.time = 0
        self.winner = -1
        self.done = False  # True after a decisive result, a draw, or truncation

    def copy(self):
        s = State.__new__(State)
        s.H, s.W = self.H, self.W
        s.army = self.army[:]
        s.owner = self.owner[:]
        s.mountain = self.mountain  # static, shared
        s.castle = self.castle[:]
        s.general = self.general[:]
        s.gpos = self.gpos  # static
        s.eliminated = self.eliminated[:]
        s.time = self.time
        s.winner = self.winner
        s.done = self.done
        return s

    # --- stats -----------------------------------------------------------
    def land(self, p):
        return self.owner.count(p)

    def total_army(self, p):
        a, o = self.army, self.owner
        return sum(a[i] for i in range(len(a)) if o[i] == p)


def from_grid(grid):
    """Build a State from a numeric grid (list of rows) as produced by the
    pinned generator after strip_neutral_castles: -2 mountain, 0 empty,
    1/2 general of player 0/1. Castle values (>2) are treated as stripped."""
    H, W = len(grid), len(grid[0])
    mountains, gpos = [], [None, None]
    for r in range(H):
        for c in range(W):
            v = int(grid[r][c])
            i = r * W + c
            if v == -2:
                mountains.append(i)
            elif v in (1, 2):
                gpos[v - 1] = i
    return State(H, W, mountains, gpos)


# --- castles ------------------------------------------------------------------

def build_cost(s, p, i):
    """Castle price for player p at cell i (35 + sum over own structures of
    max(0, 14 - 2*manhattan))."""
    W = s.W
    r, c = divmod(i, W)
    cost = BASE_COST
    o, cas, gen = s.owner, s.castle, s.general
    for j in range(len(o)):
        if o[j] == p and (cas[j] or gen[j]):
            rj, cj = divmod(j, W)
            d = abs(rj - r) + abs(cj - c)
            if d <= PROX_RADIUS:
                cost += PROX_PEN - PROX_DECAY * d
    return cost


def build_valid(s, p, action):
    kind, r, c = action[0], action[1], action[2]
    if kind != 2 or s.winner >= 0:
        return False
    if not (0 <= r < s.H and 0 <= c < s.W):
        return False
    i = r * s.W + c
    if s.owner[i] != p or s.general[i] or s.castle[i]:
        return False
    return s.army[i] >= build_cost(s, p, i)


def apply_builds(s, actions):
    """Resolve builds (player 0 then 1) and return actions with builds -> pass."""
    out = []
    for p in (0, 1):
        a = actions[p]
        if a[0] == 2:
            if build_valid(s, p, a):
                i = a[1] * s.W + a[2]
                s.army[i] -= build_cost(s, p, i)
                s.castle[i] = True
            out.append((1, 0, 0, 0, 0))
        else:
            out.append(tuple(a))
    return out


# --- moves --------------------------------------------------------------------

def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def move_order(s, actions):
    """Order of the two players' moves (pinned rule): chasing > reinforcing >
    smaller pre-step source army > lower index. Passes (kind != 0) go last."""
    H, W = s.H, s.W
    keys = []
    for p in (0, 1):
        k, si, sj, d = actions[p][0], actions[p][1], actions[p][2], actions[p][3]
        q = 1 - p
        passes = k != 0
        dr, dc = DIRS[_clip(d, 0, 3)]
        di, dj = si + dr, sj + dc
        qk, qi, qj = actions[q][0], actions[q][1], actions[q][2]
        chasing = (not passes) and qk == 0 and di == qi and dj == qj
        ci, cj = _clip(di, 0, H - 1), _clip(dj, 0, W - 1)
        reinforcing = (not passes) and s.owner[ci * W + cj] == p
        if passes:
            army = 2147483647
        else:
            army = s.army[_clip(si, 0, H - 1) * W + _clip(sj, 0, W - 1)]
        keys.append((not chasing, not reinforcing, army, p))
    return (0, 1) if keys[0] < keys[1] else (1, 0)


def move_amount(s, p, a):
    """(valid, src, dst, amount) for move action a by player p on state s."""
    k, si, sj, d, split = a
    if k == 1:
        return False, -1, -1, 0
    H, W = s.H, s.W
    if not (0 <= si < H and 0 <= sj < W):
        return False, -1, -1, 0
    dr, dc = DIRS[_clip(d, 0, 3)]
    di, dj = si + dr, sj + dc
    if not (0 <= di < H and 0 <= dj < W):
        return False, -1, -1, 0
    src, dst = si * W + sj, di * W + dj
    if s.owner[src] != p or s.eliminated[p]:
        return False, src, dst, 0
    sa = s.army[src]
    amt = sa // 2 if split == 1 else sa - 1
    if amt > sa - 1:
        amt = sa - 1
    if amt < 0:
        amt = 0
    if amt <= 0 or s.mountain[dst]:
        return False, src, dst, 0
    return True, src, dst, amt


def eliminate(s, captured, capturer):
    o, a = s.owner, s.army
    for i in range(len(o)):
        if o[i] == captured:
            a[i] = (a[i] + 1) // 2
            o[i] = capturer
            if s.general[i]:
                s.general[i] = False
                s.castle[i] = True
    s.eliminated[captured] = True
    if s.winner < 0:
        s.winner = capturer


def execute(s, p, a, spoils=True):
    """Execute one player's action in place. Returns True if it captured a general."""
    ok, src, dst, amt = move_amount(s, p, a)
    if not ok:
        return False
    o, arm = s.owner, s.army
    tgt_owner = o[dst]
    friendly = tgt_owner == p
    tgt_army = arm[dst]
    wins = amt > tgt_army
    if friendly:
        arm[dst] = tgt_army + amt
    else:
        arm[dst] = abs(tgt_army - amt)
    arm[src] -= amt
    if friendly or wins:
        o[dst] = p
    captured = wins and not friendly and s.general[dst] and tgt_owner >= 0
    if captured:
        s.general[dst] = False
        s.castle[dst] = True
        if spoils:
            eliminate(s, tgt_owner, p)
    return captured


def touches(s, p, a):
    """Deathtouch test: a valid move (kind 0) onto the opponent's starting general cell."""
    if a[0] != 0:
        return False
    ok, src, dst, amt = move_amount(s, p, a)
    return ok and dst == s.gpos[1 - p]


def global_update(s):
    t = s.time
    o, a = s.owner, s.army
    n = len(a)
    if t % 50 == 0:
        for i in range(n):
            if o[i] >= 0:
                a[i] += 1
    if t % 2 == 0:
        g, c = s.general, s.castle
        for i in range(n):
            if o[i] >= 0 and (g[i] or c[i]):
                a[i] += 1


def step(s, actions, deathtouch_turn=DEATHTOUCH_TURN, truncation=TRUNCATION):
    """Advance one turn in place with competition rules. Returns s.

    actions: two 5-sequences of ints [kind, row, col, dir, split]. Sets
    s.winner (0/1) on a decisive result; s.done on any terminal outcome
    (including draws: mutual capture/touch, or truncation)."""
    if s.done:
        return s
    acts = apply_builds(s, actions)
    if s.winner >= 0:  # cannot happen in practice; mirrors base step guard
        s.done = True
        return s

    first, second = move_order(s, acts)
    active = s.time >= deathtouch_turn

    # deathtouch / mutual-capture bookkeeping on spoils-free copies
    t_first = active and touches(s, first, acts[first])
    mid = s.copy()
    cap_first = execute(mid, first, acts[first], spoils=False)
    t_second = active and touches(mid, second, acts[second])
    after = mid.copy()
    cap_second = execute(after, second, acts[second], spoils=False)
    g_second_home = s.gpos[second]
    g_first_home = s.gpos[first]
    first_captured = s.general[g_second_home] and not mid.general[g_second_home]
    second_captured = mid.general[g_first_home] and not after.general[g_first_home]
    both_captured = first_captured and second_captured
    del cap_first, cap_second

    # base step: sequential execution with spoils
    execute(s, first, acts[first], spoils=True)
    execute(s, second, acts[second], spoils=True)
    s.time += 1

    touch = {first: t_first, second: t_second}
    both = (touch[0] and touch[1]) or both_captured
    one = (touch[0] != touch[1]) and not both_captured
    if both:
        s.winner = -1
        s.done = True
        return s
    if one:
        toucher = 0 if touch[0] else 1
        if s.winner < 0:
            eliminate(s, 1 - toucher, toucher)
        s.winner = toucher
    if s.winner >= 0:
        s.done = True
        return s
    global_update(s)
    if s.time >= truncation:
        s.done = True
    return s


# --- observations -------------------------------------------------------------

def visibility(s, p):
    H, W = s.H, s.W
    vis = [False] * (H * W)
    o = s.owner
    for i in range(H * W):
        if o[i] == p:
            r, c = divmod(i, W)
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < H:
                    base = rr * W
                    for cc in (c - 1, c, c + 1):
                        if 0 <= cc < W:
                            vis[base + cc] = True
    return vis


def observe(s, p):
    """Event-style observation dict for player p (perspective-relative owners).

    Grids are nested lists grid[r][c] of plain ints, matching the stdio
    protocol frames; keys follow the Code Bot rules §8.1."""
    H, W = s.H, s.W
    vis = visibility(s, p)
    q = 1 - p
    o, a = s.owner, s.army
    types, owners, armies = [], [], []
    my_land = my_army = opp_land = opp_army = 0
    for i in range(H * W):
        oi = o[i]
        if oi == p:
            my_land += 1
            my_army += a[i]
        elif oi == q:
            opp_land += 1
            opp_army += a[i]
    for r in range(H):
        trow, orow, arow = [], [], []
        base = r * W
        for c in range(W):
            i = base + c
            struct = s.mountain[i] or s.castle[i]
            if vis[i]:
                if s.general[i]:
                    t = T_GENERAL
                elif s.castle[i]:
                    t = T_CASTLE
                elif s.mountain[i]:
                    t = T_MOUNTAIN
                else:
                    t = T_PLAIN
                oi = o[i]
                orow.append(1 if oi == p else 2 if oi == q else 0)
                arow.append(a[i])
            else:
                t = T_STRUCT_FOG if struct else T_FOG
                orow.append(0)
                arow.append(0)
            trow.append(t)
        types.append(trow)
        owners.append(orow)
        armies.append(arow)
    return {
        "turn": s.time,
        "height": H,
        "width": W,
        "player_id": p,
        "my_land": my_land,
        "my_army": my_army,
        "opp_land": opp_land,
        "opp_army": opp_army,
        "type": types,
        "owner": owners,
        "army": armies,
    }

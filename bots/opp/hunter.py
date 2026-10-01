"""Pure-Python port of generals/agents/hunter_agent.py (generals-bots@13db8f69, MIT),
plus an optional castle-building switch (PARAMS["castles"]).

Garrison the general, feed its surplus out as one stack, advance toward the
enemy general / enemy land / farthest fog, and capture the general when able.
"""
from collections import deque

PARAMS = {"garrison": 4, "castles": False, "castle_until": 650}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bfs(H, W, passable, sources):
    INF = H * W + 5
    dist = [INF] * (H * W)
    dq = deque()
    for i in sources:
        dist[i] = 0
        dq.append(i)
    while dq:
        i = dq.popleft()
        r, c = divmod(i, W)
        nd = dist[i] + 1
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if passable[j] and dist[j] > nd:
                    dist[j] = nd
                    dq.append(j)
    return dist


def _cost(H, W, structs, i):
    r, c = divmod(i, W)
    cost = 35
    for j in structs:
        rj, cj = divmod(j, W)
        d = abs(rj - r) + abs(cj - c)
        if d <= 6:
            cost += 14 - 2 * d
    return cost


def act(obs):
    H, W = obs["height"], obs["width"]
    T = [v for row in obs["type"] for v in row]
    O = [v for row in obs["owner"] for v in row]
    A = [v for row in obs["army"] for v in row]
    n = H * W
    mine = [O[i] == 1 for i in range(n)]
    passable = [not (T[i] == 2 or T[i] == 5 or (T[i] == 3 and not mine[i])) for i in range(n)]
    gen = [i for i in range(n) if mine[i] and T[i] == 4]
    if not gen:
        return [1, 0, 0, 0, 0]
    g = gen[0]
    gen_army = A[g]
    reach = H * W
    from_gen = _bfs(H, W, passable, [g])

    if PARAMS["castles"] and obs["turn"] < PARAMS["castle_until"]:
        structs = [i for i in range(n) if mine[i] and T[i] in (3, 4)]
        best = None
        for i in range(n):
            if mine[i] and T[i] == 1 and A[i] >= 35:
                c = _cost(H, W, structs, i)
                if A[i] >= c + 2 and (best is None or c < best[0]):
                    best = (c, i)
        if best is not None:
            return [2, best[1] // W, best[1] % W, 0, 0]

    egen = [i for i in range(n) if O[i] == 2 and T[i] == 4]
    enemy = [i for i in range(n) if O[i] == 2 and T[i] != 3]
    fog = [i for i in range(n) if T[i] == 0 and passable[i] and from_gen[i] < reach]
    open_ = [i for i in range(n) if passable[i] and not mine[i] and from_gen[i] < reach]

    def farthest(cells):
        m = max(from_gen[i] for i in cells)
        return [i for i in cells if from_gen[i] == m]

    if egen:
        goal = egen
    elif enemy:
        goal = enemy
    elif fog:
        goal = farthest(fog)
    elif open_:
        goal = farthest(open_)
    else:
        return [1, 0, 0, 0, 0]
    to_goal = _bfs(H, W, passable, goal)
    INF = n + 7

    def toward(i):
        r, c = divmod(i, W)
        best_d, best_v = 0, INF
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                v = to_goal[j] if passable[j] else INF
                if v < best_v:
                    best_d, best_v = d, v
        return best_d, best_v

    egen_army = sum(A[i] for i in egen)
    best_kill = None
    best_fwd = None
    for i in range(n):
        if not mine[i] or A[i] <= 1:
            continue
        d, v = toward(i)
        adv = v < to_goal[i]
        if egen and to_goal[i] == 1 and adv and A[i] - 1 > egen_army:
            if best_kill is None or A[i] > A[best_kill[0]]:
                best_kill = (i, d)
        if i != g and adv and (best_fwd is None or A[i] > A[best_fwd[0]]):
            best_fwd = (i, d)
    if best_kill is not None:
        i, d = best_kill
        return [0, i // W, i % W, d, 0]
    gd, gv = toward(g)
    if gen_army >= 2 * PARAMS["garrison"] and gv < to_goal[g]:
        return [0, g // W, g % W, gd, 1]
    if best_fwd is not None:
        i, d = best_fwd
        return [0, i // W, i % W, d, 0]
    return [1, 0, 0, 0, 0]

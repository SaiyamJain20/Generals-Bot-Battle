from collections import deque
from heapq import heappush, heappop

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PASS = [1, 0, 0, 0, 0]
INF = 10 ** 9


class G:
    """Shared light-weight game helper (observation only)."""

    def __init__(self, obs):
        self.H = int(obs["height"])
        self.W = int(obs["width"])
        H, W = self.H, self.W
        self.n = n = H * W
        nb = []
        for i in range(n):
            r, c = divmod(i, W)
            l = []
            for d, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    l.append((rr * W + cc, d))
            nb.append(l)
        self.nb = nb
        self.mount = [False] * n
        self.ecastle = set()      # suspected/known enemy castles
        self.egen = -1
        self.g = -1
        self.t = -1
        self.last_enemy = {}      # cell -> (turn, army)
        self.my_castles = set()
        self.first = True
        self.tgt = -1
        self.dead = None

    # -- parse ---------------------------------------------------------
    def update(self, obs):
        self.t = t = int(obs["turn"])
        T = [int(v) for row in obs["type"] for v in row]
        O = [int(v) for row in obs["owner"] for v in row]
        A = [int(v) for row in obs["army"] for v in row]
        self.T, self.O, self.A = T, O, A
        n = self.n
        if self.first:
            self.first = False
            for i in range(n):
                if T[i] in (2, 5):
                    self.mount[i] = True
        self.my_castles = set()
        for i in range(n):
            ti = T[i]
            if ti == 5 and not self.mount[i]:
                self.ecastle.add(i)
            if ti == 2:
                self.mount[i] = True
            if ti == 4 and O[i] == 1:
                self.g = i
            if ti == 4 and O[i] == 2:
                self.egen = i
            if ti == 3:
                if O[i] == 1:
                    self.my_castles.add(i)
                    self.ecastle.discard(i)
                elif O[i] == 2:
                    self.ecastle.add(i)
            if O[i] == 2:
                self.last_enemy[i] = (t, A[i])
            elif O[i] == 1 and i in self.last_enemy:
                del self.last_enemy[i]
        for i in list(self.ecastle):
            if O[i] == 1:
                self.ecastle.discard(i)
        self.pas = [not self.mount[i] and T[i] != 5 for i in range(n)]
        self.mine = [O[i] == 1 for i in range(n)]
        self.my_land = int(obs["my_land"]); self.my_army = int(obs["my_army"])
        self.opp_land = int(obs["opp_land"]); self.opp_army = int(obs["opp_army"])
        self.dg = self.bfs([self.g]) if self.g >= 0 else [INF] * n
        self.structs = [i for i in range(n) if O[i] == 1 and T[i] in (3, 4)]

    # -- geometry ------------------------------------------------------
    def manh(self, a, b):
        W = self.W
        return abs(a // W - b // W) + abs(a % W - b % W)

    def bfs(self, src, pas=None):
        dist = [INF] * self.n
        dq = deque()
        for s in src:
            dist[s] = 0
            dq.append(s)
        pas = self.pas if pas is None else pas
        nb = self.nb
        while dq:
            i = dq.popleft()
            nd = dist[i] + 1
            for j, _ in nb[i]:
                if dist[j] > nd and pas[j]:
                    dist[j] = nd
                    dq.append(j)
        return dist

    def dirto(self, i, j):
        for k, d in self.nb[i]:
            if k == j:
                return d
        return -1

    def mv(self, i, j, split=0):
        d = self.dirto(i, j)
        if d < 0:
            return None
        return [0, i // self.W, i % self.W, d, split]

    def price(self, i):
        c = 35
        for j in self.structs:
            d = self.manh(i, j)
            if d <= 6:
                c += 14 - 2 * d
        return c

    # -- path ----------------------------------------------------------
    def path(self, src, dst, avoid=()):
        """Dijkstra src->dst; cost 1 per step + army of non-owned cells."""
        A, O, T, n = self.A, self.O, self.T, self.n
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
                if j != dst and (not self.pas[j] or j in avoid):
                    continue
                if j == dst and self.mount[j]:
                    continue
                if O[j] == 1:
                    c = 1
                elif T[j] == 0:
                    c = 3
                else:
                    c = 1 + A[j] + (6 if T[j] == 3 else 0)
                nd = d + c
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heappush(h, (nd, j))
        if dist[dst] >= INF:
            return None
        p = [dst]
        while p[-1] != src:
            p.append(prev[p[-1]])
        p.reverse()
        return p

    # -- threat / defense -----------------------------------------------
    def danger(self, root=None):
        """Max enemy army that could reach root soon (visible)."""
        root = self.g if root is None else root
        d = self.bfs([root])
        best = 0
        for i, o in enumerate(self.O):
            if o == 2 and self.A[i] >= 2 and d[i] <= 8:
                best = max(best, self.A[i] - (d[i] - 1))
        return best

    def pull_to(self, root, minarm=2, exclude=()):
        """One gather step: move the best own stack one step toward root over own land."""
        O, A = self.O, self.A
        par = {root: -1}
        dep = {root: 0}
        dq = deque([root])
        while dq:
            i = dq.popleft()
            for j, _ in self.nb[i]:
                if j not in par and O[j] == 1 and self.pas[j]:
                    par[j] = i
                    dep[j] = dep[i] + 1
                    dq.append(j)
        best, bk = None, None
        for i in par:
            if i == root or i in exclude or A[i] < minarm:
                continue
            k = (A[i] - 1) / dep[i]
            if bk is None or k > bk:
                bk, best = k, i
        if best is None:
            return None
        return self.mv(best, par[best])

    def defend(self, margin=2, force=False):
        """If the general is under visible threat, reinforce it."""
        g = self.g
        if g < 0:
            return None
        dg = self.danger()
        if dg <= 0:
            return None
        if self.A[g] > dg + margin and not force:
            return None
        # immediate counter: capture an adjacent attacker if possible
        best = None
        for j, _ in self.nb[g]:
            if self.O[j] == 2:
                for i, _ in self.nb[j]:
                    if self.O[i] == 1 and i != g and self.A[i] - 1 > self.A[j]:
                        if best is None or self.A[i] > self.A[best[0]]:
                            best = (i, j)
                if self.A[g] - 1 > self.A[j]:
                    if best is None:
                        best = (g, j)
        if best:
            return self.mv(*best)
        return self.pull_to(g, exclude=())

    def win_now(self):
        eg = self.egen
        if eg < 0:
            return None
        best = None
        for j, _ in self.nb[eg]:
            if self.O[j] == 1 and self.A[j] >= 2 and (self.t >= 800 or self.A[j] - 1 > self.A[eg]):
                if best is None or self.A[j] > self.A[best]:
                    best = j
        return self.mv(best, eg) if best is not None else None

    # -- expansion -----------------------------------------------------
    def expand(self, target=None, gen_keep=0, maxsend=None, enemy_ok=True, mind=None, allow_gen=True, skip=()):
        """Best capture move. target: optional distance map to bias toward."""
        O, A, T = self.O, self.A, self.T
        g = self.g
        best, bs = None, None
        bonus = (self.t % 50) >= 38
        for i in range(self.n):
            if O[i] != 1 or A[i] < 2 or i in skip:
                continue
            ig = i == g
            if ig and not allow_gen:
                continue
            for j, d in self.nb[i]:
                if not self.pas[j] or O[j] == 1:
                    continue
                if T[j] == 3 and O[j] == 0:
                    continue
                send = A[i] - 1
                split = 0
                if ig and gen_keep > 1:
                    if A[i] - A[i] // 2 >= gen_keep:
                        send, split = A[i] // 2, 1
                    else:
                        continue
                if send <= A[j]:
                    continue
                if O[j] == 2:
                    if not enemy_ok:
                        continue
                    sc = 20.0 + A[j] * 0.1 + (30 if T[j] in (3, 4) else 0)
                else:
                    sc = 10.0 * (1.5 if bonus else 1.0)
                    if T[j] == 0:
                        sc += 0.5
                    fr = sum(1 for k, _ in self.nb[j] if self.pas[k] and O[k] == 0)
                    sc += 0.4 * fr
                if target is not None:
                    sc -= 0.25 * min(target[j], 40)
                if mind is not None:
                    sc -= 0.2 * min(mind[j], 40)
                sc += 0.01 * A[i] - (0.5 if ig else 0)
                if bs is None or sc > bs:
                    bs, best = sc, [0, i // self.W, i % self.W, d, split]
        return best

    def stuck_step(self, target=None, gen_min=6):
        """Move the biggest idle stack one step toward the nearest neutral cell (or target)."""
        O, A = self.O, self.A
        g = self.g
        cand = [i for i in range(self.n) if O[i] == 1 and A[i] >= 2 and i != g]
        best = max(cand, key=lambda k: A[k]) if cand else -1
        use_g = g >= 0 and A[g] >= gen_min and (best < 0 or A[g] // 2 > A[best])
        i = g if use_g else best
        if i < 0:
            return None
        neu = [j for j in range(self.n) if self.pas[j] and O[j] == 0 and self.T[j] != 3]
        if target is not None:
            dist = target
        elif neu:
            dist = self.bfs(neu)
        else:
            return None
        bj, bv = None, INF
        for j, _ in self.nb[i]:
            if self.pas[j] and O[j] == 1 and dist[j] < bv and dist[j] < dist[i]:
                bj, bv = j, dist[j]
        return self.mv(i, bj, 1 if use_g else 0) if bj is not None else None

    def march(self, src, dst, split=0, avoid=()):
        p = self.path(src, dst, avoid)
        if not p or len(p) < 2:
            return None
        nxt = p[1]
        if self.O[nxt] != 1 and self.A[src] - 1 <= self.A[nxt]:
            return None
        return self.mv(src, nxt, split)

    def biggest(self, exclude_gen=False, near=None):
        O, A = self.O, self.A
        best, bk = -1, -1
        for i in range(self.n):
            if O[i] == 1 and A[i] > bk and not (exclude_gen and i == self.g):
                best, bk = i, A[i]
        return best

    def build(self, minprice=0, reserve=1, site_ok=None):
        """Build a castle at the cheapest own plain cell that can afford it."""
        best, bc = None, None
        for i in range(self.n):
            if self.O[i] == 1 and self.T[i] == 1 and self.A[i] >= 35:
                if site_ok is not None and not site_ok(i):
                    continue
                pr = self.price(i)
                if self.A[i] >= pr + reserve and (bc is None or pr < bc):
                    bc, best = pr, i
        if best is None:
            return None
        return [2, best // self.W, best % self.W, 0, 0]


    def pick_target(self):
        if self.egen >= 0:
            return self.egen
        if self.dead is None:
            self.dead = set()
        for i in range(self.n):
            if self.T[i] != 0 and self.T[i] != 5 and i not in self.dead and i != self.g:
                if self.tgt == i:
                    self.tgt = -1
                self.dead.add(i)
        if self.tgt >= 0 and self.tgt not in self.dead:
            return self.tgt
        r, c = divmod(self.g, self.W)
        mir = (self.H - 1 - r) * self.W + (self.W - 1 - c)
        cands = [i for i in range(self.n) if self.pas[i] and self.dg[i] < INF and self.dg[i] >= 17 and i not in self.dead]
        if not cands:
            cands = [i for i in range(self.n) if self.pas[i] and self.dg[i] < INF and i not in self.dead and i != self.g]
        if not cands:
            return -1
        self.tgt = min(cands, key=lambda i: self.manh(i, mir))
        return self.tgt



def make_act(policy):
    state = {"bot": None, "last": -1}

    def act(observation):
        try:
            t = int(observation["turn"])
            b = state["bot"]
            if b is None or t <= state["last"]:
                b = state["bot"] = policy(observation)
            state["last"] = t
            b.update(observation)
            a = b.decide()
            if a is None:
                return list(PASS)
            a = [int(x) for x in a]
            if len(a) != 5 or a[0] not in (0, 1, 2) or not 0 <= a[3] <= 3 or a[4] not in (0, 1):
                return list(PASS)
            return a
        except Exception:
            return list(PASS)
    return act


# zoo_sniper: finds enemy castles through the type-5 leak and raids them / the general with medium stacks.
class P(G):
    raider = -1
    pend = None

    def castle_est(self, c):
        if self.O[c] == 2 and self.T[c] == 3:
            return self.A[c]
        return 14

    def decide(self):
        t = self.t
        g = self.g
        a = self.win_now() or self.defend()
        if a:
            return a
        if t < 45:
            return self.expand(gen_keep=0) or self.stuck_step()
        r = self.raider
        if self.pend is not None and self.O[self.pend] == 1 and self.A[self.pend] >= 3:
            r = self.pend
        if r < 0 or self.O[r] != 1 or self.A[r] < 3:
            cand = [i for i in range(self.n) if self.O[i] == 1 and i != g and self.A[i] >= 3]
            r = max(cand, key=lambda i: self.A[i]) if cand else -1
        self.raider = r
        self.pend = None
        if r >= 0 and self.A[r] >= 10:
            d = self.bfs([r])
            tg = -1
            reach = [c for c in self.ecastle if d[c] < INF or any(d[k] < INF for k, _ in self.nb[c])]
            ok = [c for c in reach if self.A[r] - 1 > self.castle_est(c) + self.manh(r, c) // 3]
            if ok:
                tg = min(ok, key=lambda c: self.manh(r, c))
            elif self.egen >= 0 and self.A[r] > self.A[self.egen] + 8:
                tg = self.egen
            else:
                best, bk = -1, None
                for i in range(self.n):
                    if self.O[i] == 2 and d[i] < INF and self.A[r] - 1 > self.A[i] + 2:
                        k = d[i]
                        if bk is None or k < bk:
                            bk, best = k, i
                tg = best
                if tg < 0 and not self.ecastle and t > 80:
                    tg = self.pick_target()
            if tg >= 0:
                a = self.march(r, tg)
                if a:
                    dr, dc = DIRS[a[3]]
                    self.pend = (a[1] + dr) * self.W + a[2] + dc
                    return a
        # build up the raider / keep expanding
        if t % 2 == 0:
            a = self.expand(gen_keep=4, skip=(r,) if r >= 0 else ())
            if a:
                return a
        if r >= 0 and self.A[g] >= 20:
            p = self.pull_path(g, r)
            if p:
                return self.mv(g, p, 1)
        if r >= 0:
            a = self.pull_to(r, minarm=2, exclude=(g,))
            if a:
                return a
        return self.expand(gen_keep=4) or self.stuck_step()

    def pull_path(self, s, d):
        par = {s: -1}
        dq = deque([s])
        while dq:
            i = dq.popleft()
            if i == d:
                break
            for j, _ in self.nb[i]:
                if j not in par and self.O[j] == 1:
                    par[j] = i
                    dq.append(j)
        if d not in par:
            return None
        x = d
        while par[x] != s:
            x = par[x]
            if x == -1:
                return None
        return x


act = make_act(P)


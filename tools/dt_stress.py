"""Deathtouch stress: from turn START the opponent becomes an oracle that marches
its biggest stacks straight at our (known) general and touches at >= 800.
Reports how often our bot survives until turn END."""
import json
import sys
from collections import deque
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load

START, END = 770, 900


def bfs(s, src):
    W = s.W
    d = [10 ** 9] * (s.H * W)
    d[src] = 0
    q = deque([src])
    while q:
        i = q.popleft()
        r, c = divmod(i, W)
        for dr, dc in E.DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < s.H and 0 <= cc < W:
                j = rr * W + cc
                if not s.mountain[j] and d[j] > d[i] + 1:
                    d[j] = d[i] + 1
                    q.append(j)
    return d


def oracle(s, p):
    tgt = s.gpos[1 - p]
    d = bfs(s, tgt)
    W = s.W
    own = [i for i in range(s.H * W) if s.owner[i] == p and s.army[i] >= 2]
    if not own:
        return [1, 0, 0, 0, 0]
    # touch if possible
    if s.time >= 800:
        for i in own:
            if d[i] == 1:
                r, c = divmod(i, W)
                for k, (dr, dc) in enumerate(E.DIRS):
                    if (r + dr) * W + c + dc == tgt:
                        return [0, r, c, k, 0]
    i = max(own, key=lambda k: s.army[k] - 0.5 * d[k])
    r, c = divmod(i, W)
    best = None
    for k, (dr, dc) in enumerate(E.DIRS):
        rr, cc = r + dr, c + dc
        if 0 <= rr < s.H and 0 <= cc < W:
            j = rr * W + cc
            if d[j] < d[i] and (s.owner[j] == p or s.army[i] - 1 > s.army[j]) and (j != tgt or s.time >= 800 or s.army[i] - 1 > s.army[j]):
                best = [0, r, c, k, 0]
    return best or [1, 0, 0, 0, 0]


def one(args):
    a, b, k = args
    mp = [json.loads(l) for _, l in zip(range(k + 1), open("data/maps.jsonl"))][k]
    s = E.from_grid(mp["grid"])
    bots = [load(a, "a%d" % k), load(b, "b%d" % k)]
    while not s.done and s.time < END:
        acts = [bots[0].act(E.observe(s, 0)), bots[1].act(E.observe(s, 1))]
        if s.time >= START:
            acts[1] = oracle(s, 1)
        E.step(s, acts)
    return s.winner, s.time


if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(4) as p:
        res = p.map(one, [(a, b, k) for k in range(n)])
    lost_late = sum(1 for w, t in res if w == 1 and t > START)
    lost_early = sum(1 for w, t in res if w == 1 and t <= START)
    won = sum(1 for w, t in res if w == 0)
    print("won", won, "lost before start", lost_early, "lost to oracle", lost_late, "survived", n - won - lost_early - lost_late)

"""When do we first see the enemy general, and how do games end? findstats.py A B n"""
import json
import statistics as st
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load


def one(args):
    a, b, k = args
    mp = [json.loads(l) for _, l in zip(range(k // 2 + 1), open("data/maps.jsonl"))][k // 2]
    s = E.from_grid(mp["grid"])
    swap = k % 2 == 1
    paths = (b, a) if swap else (a, b)
    bots = [load(paths[0], "a%d" % k), load(paths[1], "b%d" % k)]
    me = 1 if swap else 0
    seen = -1
    ncand = []
    while not s.done:
        acts = [bots[p].act(E.observe(s, p)) for p in (0, 1)]
        bt = bots[me]._BOT
        if seen < 0 and bt.egen >= 0:
            seen = s.time
        if s.time in (100, 200, 300, 400):
            ncand.append(len(bt.cands))
        E.step(s, acts)
    return s.winner == me, s.winner < 0, s.time, seen, ncand


if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 5) as p:
        res = p.map(one, [(a, b, k) for k in range(n)])
    seen = [r[3] for r in res if r[3] >= 0]
    print("win", sum(r[0] for r in res), "draw", sum(r[1] for r in res), "n", n,
          "median end", st.median(r[2] for r in res),
          "seen frac", round(len(seen) / n, 2), "median seen", st.median(seen) if seen else None)
    for t, idx in ((100, 0), (200, 1), (300, 2), (400, 3)):
        v = [r[4][idx] for r in res if len(r[4]) > idx]
        if v:
            print("cands at", t, "median", st.median(v))

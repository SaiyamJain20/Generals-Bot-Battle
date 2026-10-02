"""Economy curves (land, army, castles) of A vs B at fixed turns: econ.py A B n workers"""
import json
import statistics as st
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load

TS = (100, 150, 200, 250, 300, 400)


def one(args):
    a, b, k = args
    mp = [json.loads(l) for _, l in zip(range(k + 1), open("data/maps.jsonl"))][k]
    s = E.from_grid(mp["grid"])
    bots = [load(a, "a%d" % k), load(b, "b%d" % k)]
    out = {}
    while not s.done and s.time <= max(TS):
        E.step(s, [bots[0].act(E.observe(s, 0)), bots[1].act(E.observe(s, 1))])
        if s.time in TS:
            out[s.time] = [(s.land(p), s.total_army(p), sum(1 for i, c in enumerate(s.castle) if c and s.owner[i] == p))
                           for p in (0, 1)]
    return out, s.winner


if __name__ == "__main__":
    a, b, n, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    with Pool(w) as p:
        res = p.map(one, [(a, b, k) for k in range(n)])
    print("A wins", sum(1 for _, wnr in res if wnr == 0), "of", n)
    for t in TS:
        v = [r[t] for r, _ in res if t in r]
        if v:
            f = lambda p, k: st.median(x[p][k] for x in v)
            print(t, "n", len(v), "A land/army/castles", f(0, 0), f(0, 1), f(0, 2), "| B", f(1, 0), f(1, 1), f(1, 2))

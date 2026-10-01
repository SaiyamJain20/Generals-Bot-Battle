"""Rank of the true enemy general under our belief scores at fixed turns."""
import json
import statistics as st
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load

TURNS = (60, 100, 140, 180, 240)


def one(args):
    a, b, k = args
    mp = [json.loads(l) for _, l in zip(range(k // 2 + 1), open("data/maps.jsonl"))][k // 2]
    s = E.from_grid(mp["grid"])
    swap = k % 2 == 1
    paths = (b, a) if swap else (a, b)
    bots = [load(paths[0], "a%d" % k), load(paths[1], "b%d" % k)]
    me = 1 if swap else 0
    out = {}
    while not s.done and s.time <= max(TURNS):
        acts = [bots[p].act(E.observe(s, p)) for p in (0, 1)]
        if s.time in TURNS:
            bt = bots[me]._BOT
            true = s.gpos[1 - me]
            if bt.egen >= 0:
                out[s.time] = (1, 1, True)
            else:
                sc = bt.belief_scores(None)
                order = sorted(sc, key=sc.get)
                rank = order.index(true) + 1 if true in order else None
                out[s.time] = (rank, len(order), False)
        E.step(s, acts)
    return out


if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(5) as p:
        res = p.map(one, [(a, b, k) for k in range(n)])
    for t in TURNS:
        v = [r[t] for r in res if t in r]
        known = sum(1 for x in v if x[2])
        ranks = [x[0] for x in v if not x[2] and x[0] is not None]
        missing = sum(1 for x in v if not x[2] and x[0] is None)
        sizes = [x[1] for x in v if not x[2]]
        print(t, "known", known, "n", len(v), "true-not-in-cands", missing,
              "median rank", st.median(ranks) if ranks else None,
              "top3", sum(1 for r in ranks if r <= 3), "median cands", st.median(sizes) if sizes else None)

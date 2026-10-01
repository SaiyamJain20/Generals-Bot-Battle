"""Land at turns 50 and 100 for a bot playing alone (opponent passes)."""
import json
import statistics as st
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load


def one(args):
    path, k, params = args
    mp = [json.loads(l) for _, l in zip(range(k + 1), open("data/maps.jsonl"))][k]
    s = E.from_grid(mp["grid"])
    b = load(path, "o%d" % k)
    if params:
        b.PARAMS.update(params)
    out = {}
    while s.time < 101:
        a = b.act(E.observe(s, 0))
        E.step(s, [a, [1, 0, 0, 0, 0]])
        if s.time in (50, 100):
            out[s.time] = s.land(0)
    return out


if __name__ == "__main__":
    path, n = sys.argv[1], int(sys.argv[2])
    params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    with Pool(5) as p:
        res = p.map(one, [(path, k, params) for k in range(n)])
    for t in (50, 100):
        v = [r[t] for r in res]
        print(t, "mean", round(st.mean(v), 2), "min", min(v), "max", max(v))

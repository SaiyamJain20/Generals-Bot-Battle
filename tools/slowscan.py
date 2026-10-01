"""Find slow turns: python tools/slowscan.py botA botB ngames threshold_ms"""
import json
import sys
import time
from multiprocessing import Pool
sys.path.insert(0, ".")
from sim import engine as E
from tools.trace import load

MAPS = None


def one(args):
    a, b, k, thr = args
    global MAPS
    if MAPS is None:
        MAPS = [json.loads(l) for _, l in zip(range(500), open("data/maps.jsonl"))]
    s = E.from_grid(MAPS[k // 2]["grid"])
    swap = k % 2 == 1
    paths = (b, a) if swap else (a, b)
    bots = [load(paths[0], "x%d" % k), load(paths[1], "y%d" % k)]
    me = 1 if swap else 0
    out = []
    while not s.done:
        acts = []
        for p in (0, 1):
            o = E.observe(s, p)
            t0 = time.process_time()
            acts.append(bots[p].act(o))
            dt = (time.process_time() - t0) * 1000
            if p == me and s.time > 0 and dt > thr:
                bt = bots[p]._BOT
                out.append((round(dt, 1), k, s.time, bt.last_label, (bt.cyc or {}).get("mode"),
                            bt.my_land, len(bt.my_castles)))
        E.step(s, acts)
    return out


if __name__ == "__main__":
    a, b, n, thr = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    allr = []
    with Pool(15) as pool:
        for r in pool.imap_unordered(one, [(a, b, k, thr) for k in range(n)]):
            allr.extend(r)
    allr.sort(reverse=True)
    print(len(allr), "slow turns")
    for x in allr[:25]:
        print(x)

"""Training data for the learned threat model (garrison sizing).

Replays (with inferred actions) are re-simulated exactly; from each player's
perspective our own Bot.update() builds its memory, and we record features of
the fogged situation together with the TRUE future threat: the largest enemy
army standing within BFS distance R of our general during the next HORIZON
turns.

    python learn/threat_data.py --games 600 --out data/threat.jsonl
"""
import argparse
import gzip
import importlib.util
import json
import math
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

R, HORIZON, STRIDE = 6, 12, 3
RDIR = os.path.join(ROOT, "data", "replays")
_MOD = None


def mod():
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("pb_threat", os.path.join(ROOT, "bots", "participant.py"))
        _MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MOD)
    return _MOD


def features(b):
    hidden = b.opp_army - b.vis_enemy_army - max(0, b.opp_land - b.vis_enemy_cells - 1)
    vis_max = max((b.A[i] for i in range(b.n) if b.O[i] == 2), default=0)
    tr_army = max((tr[1] for tr in b.tracks), default=0)
    tr_dist = min((max(1, b.dist_g[tr[0]] - (b.turn - tr[2])) for tr in b.tracks), default=40)
    fd = b.fog_distance()
    return [math.log1p(max(0, hidden)), math.log1p(b.opp_army), math.log1p(b.my_army),
            b.opp_land / 100.0, b.my_land / 100.0, min(fd, 20) / 10.0, len(b.enemy_castles) / 5.0,
            b.turn / 1000.0, math.log1p(vis_max), math.log1p(tr_army), min(tr_dist, 40) / 10.0,
            math.log1p(b.vis_enemy_army)]


def one(rec):
    m = mod()
    d = json.loads(gzip.open(os.path.join(RDIR, rec["file"])).read())
    H, W = d["dims"]["rows"], d["dims"]["cols"]
    s = E.State(H, W, [r * W + c for r, c in d["mountains"]], [r * W + c for r, c in d["generals"]])
    bots = {}
    dist = {}
    for p in (0, 1):
        dist[p] = None
    rows = {0: [], 1: []}
    near = {0: [], 1: []}
    for t, acts in enumerate(rec["acts"]):
        if acts is None:
            break
        for p in (0, 1):
            obs = m._normalize(E.observe(s, p))
            if t == 0:
                b = m.Bot(obs)
                bots[p] = b
            b = bots[p]
            T, O, A = b.parse(obs)
            b.turn = t
            if b.general < 0:
                b.general = s.gpos[p]
                b.setup(T)
                dist[p] = b.dist_g
            b.update(obs, T, O, A)
            b.last_action = list(acts[p])
            # true max enemy army within R of our general (this turn)
            dg = dist[p]
            q = 1 - p
            near[p].append(max((s.army[i] for i in range(H * W) if s.owner[i] == q and dg[i] <= R), default=0))
            if t >= 50 and t % STRIDE == 0:
                rows[p].append((t, features(b)))
        E.step(s, acts)
    out = []
    for p in (0, 1):
        series = near[p]
        for t, f in rows[p]:
            fut = series[t + 1:t + 1 + HORIZON]
            if len(fut) < HORIZON:
                continue
            out.append({"f": f, "y": math.log1p(max(fut)), "t": t})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data/threat.jsonl")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "acts.jsonl"))]
    recs = [r for r in recs if r["fails"] == 0]
    random.Random(0).shuffle(recs)
    recs = recs[: args.games]
    n = 0
    with Pool(args.workers) as pool, open(args.out, "w") as f:
        for rows in pool.imap_unordered(one, recs, chunksize=2):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n += 1
    print("rows", n, file=sys.stderr)


if __name__ == "__main__":
    main()

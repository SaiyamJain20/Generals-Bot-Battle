"""Validate enemy-castle accounting against real replays (true builds known)."""
import gzip
import importlib.util
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("pb_acc", os.path.join(ROOT, "bots", "participant.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def run(rec):
    d = json.loads(gzip.open(os.path.join(ROOT, "data", "replays", rec["file"])).read())
    H, W = d["dims"]["rows"], d["dims"]["cols"]
    s = E.State(H, W, [r * W + c for r, c in d["mountains"]], [r * W + c for r, c in d["generals"]])
    bots = {}
    true_builds = {0: [], 1: []}
    ring_bad = 0
    for t, acts in enumerate(rec["acts"]):
        if acts is None:
            break
        for p in (0, 1):
            obs = M._normalize(E.observe(s, p))
            if t == 0:
                bots[p] = M.Bot(obs)
            b = bots[p]
            T, O, A = b.parse(obs)
            b.turn = t
            if b.general < 0:
                b.general = s.gpos[p]
                b.setup(T)
            b.update(obs, T, O, A)
            b.last_action = list(acts[p])
            # true enemy general must stay in candidates
            if s.gpos[1 - p] not in b.cands:
                ring_bad += 1
        pre = s.copy()
        for p in (0, 1):
            a = acts[p]
            if a[0] == 2 and E.build_valid(pre, p, a):
                i = a[1] * W + a[2]
                true_builds[p].append((t + 1, i, E.build_cost(pre, p, i)))
        E.step(s, acts)
    res = {}
    for p in (0, 1):
        det = [(tt, c, pr) for tt, c, pr in bots[p].enemy_builds]
        tru = true_builds[1 - p]
        res[p] = (len(tru), len(det), len(set(det) & set(tru)))
    return res, ring_bad


if __name__ == "__main__":
    recs = [json.loads(l) for _, l in zip(range(400), open(os.path.join(ROOT, "data", "acts.jsonl")))]
    recs = [r for r in recs if r["fails"] == 0][: int(sys.argv[1]) if len(sys.argv) > 1 else 15]
    T = D = Mt = RB = 0
    for r in recs:
        res, rb = run(r)
        for p, (t, d, m) in res.items():
            T += t
            D += d
            Mt += m
        RB += rb
        print(r["file"][:30], res, "true-general-pruned-turns", rb, flush=True)
    print(f"true builds {T}, detected {D}, exact matches (turn, cell, price) {Mt}, pruned-true-general turn-count {RB}")

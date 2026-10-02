"""Compare fog-stack estimate vs true largest hidden enemy stack on replays."""
import gzip
import importlib.util
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("pb_fog", os.path.join(ROOT, "bots", "participant.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

rows = []
recs = [json.loads(l) for _, l in zip(range(60), open(os.path.join(ROOT, "data", "acts.jsonl")))]
for rec in [r for r in recs if r["fails"] == 0][: int(sys.argv[1]) if len(sys.argv) > 1 else 8]:
    d = json.loads(gzip.open(os.path.join(ROOT, "data", "replays", rec["file"])).read())
    H, W = d["dims"]["rows"], d["dims"]["cols"]
    s = E.State(H, W, [r * W + c for r, c in d["mountains"]], [r * W + c for r, c in d["generals"]])
    bots = {}
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
            if t >= 60 and t % 5 == 0:
                vis = E.visibility(s, p)
                q = 1 - p
                true = max((s.army[i] for i in range(H * W) if s.owner[i] == q and not vis[i]
                            and i != s.gpos[q]), default=0)
                hidden = b.opp_army - b.vis_enemy_army - max(0, b.opp_land - b.vis_enemy_cells - 1)
                rows.append((true, b.fog_risk(0), b.fog_risk(5), hidden))
        E.step(s, acts)


def stats(k):
    err = [abs(math.log1p(r[k]) - math.log1p(r[0])) for r in rows]
    over = sum(1 for r in rows if r[k] >= r[0]) / len(rows)
    return round(sum(err) / len(err), 3), round(over, 3)


print("n", len(rows))
for name, k in (("fog_risk(0)", 1), ("fog_risk(5)", 2), ("hidden bound", 3)):
    print(name, "mean |log err|", stats(k)[0], "covers true (>=)", stats(k)[1])
for f in (0.3, 0.5):
    err = [abs(math.log1p(r[3] * f) - math.log1p(r[0])) for r in rows]
    print(f"hidden*{f}", round(sum(err) / len(err), 3), round(sum(1 for r in rows if r[3] * f >= r[0]) / len(rows), 3))

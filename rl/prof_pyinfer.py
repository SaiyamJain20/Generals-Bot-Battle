"""Per-stage timing on recorded positions: python rl/prof_pyinfer.py NET [SA KS KH]  (NET e.g. 12x1, 8x2, 16x1 (random weights))"""
import os
import pickle
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "rl"), ROOT]
import torch  # noqa: E402
import pack_student as PK  # noqa: E402
import pyfeat as PF  # noqa: E402
import pyinfer as PI  # noqa: E402

net = sys.argv[1]
SA, KS, KH = (int(x) for x in sys.argv[2:5]) if len(sys.argv) >= 5 else (8, 14, 10)
obss = pickle.load(open("/tmp/rl_deploy_pos.pkl", "rb"))
path = os.path.join(ROOT, f"rl/students/ResBot_{net}.pt")
if os.path.exists(path):
    ck = torch.load(path, map_location="cpu")
else:  # synthetic weights, only for timing
    import learn.bc_train as BT
    ch, bl = (int(x) for x in net.split("x"))
    m = BT.Net(ch=ch, blocks=bl)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2)
            mod.running_mean.normal_(0, 0.1)
    ck = {"ch": ch, "blocks": bl, "state": m.state_dict()}
pn = PI.Net(PK.pack(ck, "e"), SA, KS, KH)
REPS = int(os.environ.get("REPS", "3"))
runs = []
for rep in range(REPS):
    trk = None
    row = []
    for obs in obss:
        H, W = obs["height"], obs["width"]
        if trk is None or obs["turn"] == 0:
            trk = PF.Tracker(H, W)
        t0 = time.perf_counter()
        P, sc, (O, A, T) = trk.update(obs)
        legal = PF.legal_moves(O, A, T, H, W, P)
        t1 = time.perf_counter()
        pn.policy(P, sc, legal, H, W)
        t2 = time.perf_counter()
        row.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
    runs.append(row)


def pct(v):
    v = sorted(v)
    return v[len(v) // 2], v[int(len(v) * .99)], v[-1]


raw = [a + b for row in runs for a, b in row]
best = [min(runs[r][i][0] + runs[r][i][1] for r in range(REPS)) for i in range(len(obss))]
print("%s: feat mean %.2f, infer mean %.2f | raw (loaded laptop) p50/p99/max %.2f/%.2f/%.2f | best-of-%d per position p50/p99/max %.2f/%.2f/%.2f ms (n=%d)" % (
    net, sum(a for row in runs for a, b in row) / len(raw), sum(b for row in runs for a, b in row) / len(raw),
    *pct(raw), REPS, *pct(best), len(raw)))

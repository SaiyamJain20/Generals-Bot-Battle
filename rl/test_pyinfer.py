"""Compare pyinfer.Net (blob mode, fixed-point params) with the torch student: argmax agreement + timing.
   python rl/test_pyinfer.py NET(12x1|8x2|16x1) MODE(e|f|q) SA KS KH GAMES MAXT
"""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "learn"), os.path.join(ROOT, "rl")]
import numpy as np, torch
import arena.run as R
from sim import engine as E
import pyfeat as PF, pyinfer as PI, pack_student as PK
from bc_train import Net

def main():
    net, mode = sys.argv[1], sys.argv[2]
    SA, KS, KH = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    ng, maxt = int(sys.argv[6]), int(sys.argv[7])
    torch.set_num_threads(1)
    ck = torch.load(os.path.join(ROOT, f"rl/students/ResBot_{net}.pt"), map_location="cpu")
    tn = Net(ch=ck["ch"], blocks=ck["blocks"]); tn.load_state_dict(ck["state"]); tn.eval()
    pn = PI.Net(PK.pack(ck, mode), SA, KS, KH)
    opp = os.path.join(ROOT, "bots/opp/expander.py")
    agree = tot = 0; times = []; maxdl = 0.0; top_ties = 0
    for g in range(ng):
        m = R.maps()[g * 11 + 3]
        s = E.from_grid(m["grid"])
        mod = R.load_bot(os.path.join(ROOT, f"rl/bots/student_{net}.py")) if os.path.exists(os.path.join(ROOT, f"rl/bots/student_{net}.py")) else R.load_bot(os.path.join(ROOT, "rl/bots/student_12x1.py"))
        ex = R.load_bot(opp)
        H, W = s.H, s.W
        trk = PF.Tracker(H, W)
        side = g % 2
        while not s.done and s.time < maxt:
            acts = [None, None]
            for p in (0, 1):
                obs = E.observe(s, p)
                if p == side:
                    P, sc, (O, A, T) = trk.update(obs)
                    legal = PF.legal_moves(O, A, T, H, W, P)
                    t0 = time.perf_counter()
                    idx, _ = pn.policy(P, sc, legal, H, W)
                    times.append((time.perf_counter() - t0) * 1000)
                    # torch reference
                    pl = np.array(PF.dense(P, H, W), np.uint8).reshape(1, 16, 21, 21)
                    with torch.no_grad():
                        lg, _ = tn(torch.from_numpy(pl), torch.tensor([sc], dtype=torch.float32))
                    lg = lg[0].numpy().copy()
                    mask = np.zeros(PF.NACT, bool); mask[legal] = True; mask[-1] = True
                    lg[~mask] = -1e9
                    ref = int(lg.argmax())
                    agree += (ref == idx); tot += 1
                    acts[p] = PF.index_action(ref)
                else:
                    acts[p] = (ex if True else None).act(obs)
            E.step(s, acts)
    ts = sorted(times)
    print(f"{net} mode {mode} SA{SA} KS{KS} KH{KH}: agree {agree}/{tot} = {agree/tot:.4f}  ms p50 {ts[len(ts)//2]:.2f} p99 {ts[int(len(ts)*.99)]:.2f} max {ts[-1]:.2f}")

main()

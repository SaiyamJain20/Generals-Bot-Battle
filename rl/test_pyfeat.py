"""Parity of rl/pyfeat.py vs learn/bc_features.py over real games (torch student vs expander)."""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "learn"), os.path.join(ROOT, "rl")]
import numpy as np
import arena.run as R
from sim import engine as E
import bc_features as F
import pyfeat as PF

def main(ngames=20, maxturns=int(os.environ.get("MAXT", "1200"))):
    bots = [os.path.join(ROOT, "rl/bots/student_12x1.py"), os.path.join(ROOT, "bots/opp/expander.py")]
    maxd_pl = 0; maxd_sc = 0.0; npos = 0; nlegal_bad = 0; nidx_bad = 0
    for g in range(ngames):
        m = R.maps()[g * 7]
        s = E.from_grid(m["grid"])
        mods = [R.load_bot(b) for b in (bots if g % 2 == 0 else bots[::-1])]
        H, W = s.H, s.W
        trk_np = [F.Tracker(H, W), F.Tracker(H, W)]
        trk_py = [PF.Tracker(H, W), PF.Tracker(H, W)]
        while not s.done and s.time < maxturns:
            acts = []
            for p in (0, 1):
                obs = E.observe(s, p)
                acts.append(mods[p].act(obs))
                pl, sc, (O, A, T) = trk_np[p].update(obs)
                P, scp, (O2, A2, T2) = trk_py[p].update(obs)
                d = np.array(PF.dense(P, H, W), np.int32).reshape(16, 21, 21)
                maxd_pl = max(maxd_pl, int(np.abs(d - pl.astype(np.int32)).max()))
                maxd_sc = max(maxd_sc, float(np.abs(np.array(scp, np.float64) - sc.astype(np.float64)).max()))
                mask = F.legal_mask(O, A, T, H, W, pl)
                ref = [int(i) for i in np.nonzero(mask[:-1])[0]]
                if ref != PF.legal_moves(O2, A2, T2, H, W, P):
                    nlegal_bad += 1
                for i in ref[:5]:
                    if PF.action_index(PF.index_action(i)) != i or F.index_action(i) != PF.index_action(i):
                        nidx_bad += 1
                npos += 1
            E.step(s, acts)
        print(f"game {g} turns {s.time} maxdiff planes {maxd_pl} scal {maxd_sc:.2e}", flush=True)
    print(f"positions {npos} max plane diff {maxd_pl} max scalar diff {maxd_sc:.3e} legal mismatches {nlegal_bad} idx mismatches {nidx_bad}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

"""Timing of pyfeat + pyinfer on recorded positions (single core: run under taskset).
   python rl/bench_pyinfer.py NET MODE SA KS KH [reps]"""
import os, sys, time, pickle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "rl")]
import pyfeat as PF, pyinfer as PI
import torch, pack_student as PK

POS = "/tmp/rl_deploy_pos.pkl"

def record():
    import arena.run as R
    from sim import engine as E
    opp = os.path.join(ROOT, "bots/opp/expander.py")
    pos = []
    for g in range(6):
        m = R.maps()[g * 5 + 1]; s = E.from_grid(m["grid"])
        mods = [R.load_bot(opp), R.load_bot(os.path.join(ROOT, "bots/opp/ext_a9.py"))]
        H, W = s.H, s.W; trk = PF.Tracker(H, W)
        while not s.done and s.time < 900:
            acts = []
            for p in (0, 1):
                obs = E.observe(s, p)
                acts.append(mods[p].act(obs))
                if p == 0 and s.time % 6 == 0:
                    pos.append(obs)
                if p == 0:
                    pass
            E.step(s, acts)
        print("game", g, s.time, flush=True)
    pickle.dump(pos, open(POS, "wb"))

def main():
    if not os.path.exists(POS):
        record()
    obss = pickle.load(open(POS, "rb"))
    net, mode, SA, KS, KH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    ck = torch.load(os.path.join(ROOT, f"rl/students/ResBot_{net}.pt"), map_location="cpu")
    pn = PI.Net(PK.pack(ck, mode), SA, KS, KH)
    # tracker needs sequential updates: replay positions grouped by game (turn resets)
    tf = []; ti = []
    for rep in range(reps):
        trk = None
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
            tf.append((t1 - t0) * 1000); ti.append((t2 - t1) * 1000)
    tot = sorted(a + b for a, b in zip(tf, ti)); n = len(tot)
    print(f"{net} {mode} SA{SA}: n={n} feat mean {sum(tf)/n:.2f} infer mean {sum(ti)/n:.2f} | total p50 {tot[n//2]:.2f} p99 {tot[int(n*.99)]:.2f} max {tot[-1]:.2f} ms")

main()

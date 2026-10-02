"""Action parity pystudent.py vs torch student_12x1.py on the same positions (torch bot drives the game)."""
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT]
import arena.run as R
from sim import engine as E

def main(ng, maxt, ref_name="student_12x1.py", opp_name="bots/opp/expander.py"):
    opp = os.path.join(ROOT, opp_name)
    same = tot = 0; times = []; first = []
    for g in range(ng):
        s = E.from_grid(R.maps()[g * 13 + 5]["grid"])
        side = g % 2
        t0 = time.perf_counter()
        py = R.load_bot(os.path.join(ROOT, "rl/bots/pystudent.py"))
        first.append((time.perf_counter() - t0) * 1000)
        ref = R.load_bot(os.path.join(ROOT, "rl/bots", ref_name))
        ex = R.load_bot(opp)
        while not s.done and s.time < maxt:
            acts = [None, None]
            for p in (0, 1):
                obs = E.observe(s, p)
                if p == side:
                    t1 = time.perf_counter(); a = py.act(obs); times.append((time.perf_counter() - t1) * 1000)
                    b = ref.act(obs)
                    same += (a == b); tot += 1
                    acts[p] = b
                else:
                    acts[p] = ex.act(obs)
            E.step(s, acts)
        print(f"game {g} turns {s.time} so far {same}/{tot}", flush=True)
    ts = sorted(times); n = len(ts)
    print(f"RESULT same-move rate {same}/{tot} = {same/tot:.4f}; pystudent act ms p50 {ts[n//2]:.2f} p99 {ts[int(n*.99)]:.2f} max {ts[-1]:.2f}; import ms {sum(first)/len(first):.1f}")

main(int(sys.argv[1]), int(sys.argv[2]))

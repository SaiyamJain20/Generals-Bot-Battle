"""Realistic match runner: each bot in its own process, both pinned to ONE CPU
core (the event shares one CPU between alternating bots), JSON over pipes, and
the event's limits enforced on wall-clock reply time (150 ms, first 10 s).

    python arena/subproc.py A.py B.py --games 20 [--limit-ms 150] [--slowdown 1.0] [--core 3]

--slowdown N runs a CPU-burning helper on the same core so each bot gets
roughly 1/N of it (a crude model of a slower event machine).
"""
import argparse
import json
import os
import select
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

CHILD = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("bot", sys.argv[1])
mod = None
for line in sys.stdin:
    obs = json.loads(line)
    try:
        if mod is None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        a = mod.act(obs)
        out = json.dumps([int(v) for v in a]) if isinstance(a, (list, tuple)) else "null"
    except Exception as e:
        out = json.dumps({"error": repr(e)})
    sys.stdout.write(out + "\n")
    sys.stdout.flush()
"""


def spawn(path, core):
    p = subprocess.Popen([sys.executable, "-c", CHILD, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        os.sched_setaffinity(p.pid, {core})
    except Exception:
        pass
    return p


def burner(core):
    p = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        os.sched_setaffinity(p.pid, {core})
    except Exception:
        pass
    return p


def ask(proc, obs, limit_s):
    t0 = time.perf_counter()
    proc.stdin.write(json.dumps(obs) + "\n")
    proc.stdin.flush()
    r, _, _ = select.select([proc.stdout], [], [], limit_s)
    dt = time.perf_counter() - t0
    if not r:
        return None, dt, "timeout"
    line = proc.stdout.readline()
    try:
        a = json.loads(line)
    except Exception:
        return None, dt, "malformed"
    if not (isinstance(a, list) and len(a) == 5 and a[0] in (0, 1, 2) and 0 <= a[3] <= 3 and a[4] in (0, 1)):
        return None, dt, "malformed:" + line.strip()[:80]
    return a, dt, None


def play(paths, grid, core, limit_ms, slowdown):
    s = E.from_grid(grid)
    procs = [spawn(paths[0], core), spawn(paths[1], core)]
    burners = [burner(core) for _ in range(max(0, int(round(slowdown)) - 1))]
    times = [[], []]
    res = {"winner": -1, "reason": "draw"}
    try:
        while not s.done:
            acts = []
            for p in (0, 1):
                lim = (10000 if s.time == 0 else limit_ms) / 1000.0
                a, dt, err = ask(procs[p], E.observe(s, p), lim)
                times[p].append(dt * 1000)
                if err:
                    res = {"winner": 1 - p, "reason": f"forfeit_{err}_p{p}", "turn": s.time}
                    return res, times
                acts.append(a)
            E.step(s, acts)
        res = {"winner": s.winner, "reason": "win" if s.winner >= 0 else "draw", "turn": s.time}
        return res, times
    finally:
        for pr in procs + burners:
            pr.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--limit-ms", type=float, default=150)
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--core", type=int, default=os.cpu_count() - 1)
    ap.add_argument("--map-offset", type=int, default=500)
    args = ap.parse_args()
    maps = [json.loads(l) for _, l in zip(range(args.map_offset + args.games), open(os.path.join(ROOT, "data", "maps.jsonl")))]
    tally = {"A": 0, "B": 0, "D": 0}
    worst = [0.0, 0.0]
    for g in range(args.games):
        swap = g % 2 == 1
        paths = (args.b, args.a) if swap else (args.a, args.b)
        res, times = play(paths, maps[args.map_offset + g // 2]["grid"], args.core, args.limit_ms, args.slowdown)
        a_player = 1 if swap else 0
        if res["winner"] < 0:
            tally["D"] += 1
        elif res["winner"] == a_player:
            tally["A"] += 1
        else:
            tally["B"] += 1
        ta = sorted(times[a_player][1:]) or [0]
        worst[0] = max(worst[0], ta[-1])
        worst[1] = max(worst[1], times[a_player][0] if times[a_player] else 0)
        print(json.dumps({"game": g, **res, "A_max_ms": round(ta[-1], 1),
                          "A_p99_ms": round(ta[int(0.99 * (len(ta) - 1))], 1),
                          "A_first_ms": round(times[a_player][0], 1) if times[a_player] else None}), flush=True)
    print(json.dumps({"tally": tally, "A_worst_ms": round(worst[0], 1), "A_worst_first_ms": round(worst[1], 1)}))


if __name__ == "__main__":
    main()

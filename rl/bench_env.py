"""Throughput of the per-game Python generator (sim/engine.py + learn/bc_features.py), 1 core.

    taskset -c 2 env OMP_NUM_THREADS=1 PYTHONPATH=vendor/generals-bots:. \
        .venv312/bin/python rl/bench_env.py --games 3

A "step" = both players observe + featurise + act once, then engine.step.
Actions: sampled from the legal mask with weight ~ source army (expands like a
weak bot, so boards fill up roughly like real games). Prints per-component us.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "learn"))
from sim import engine as E  # noqa: E402
import bc_features as F  # noqa: E402


def load_maps(n, path=os.path.join(ROOT, "data", "maps.jsonl")):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def army_weighted_action(mask, A, rng):
    """Legal action with prob ~ source army (moves only, small pass prob)."""
    idx = np.flatnonzero(mask[:F.PASS_IDX])
    if len(idx) == 0 or rng.random() < 0.02:
        return F.PASS_IDX
    cell = idx // 9
    r, c = cell // F.S, cell % F.S
    w = A[r, c].astype(np.float64) ** 2
    return int(idx[np.searchsorted(np.cumsum(w), rng.random() * w.sum())])


def run(games, max_turns, seed=0):
    rng = np.random.default_rng(seed)
    maps = load_maps(games)
    tm = {"observe": 0.0, "features": 0.0, "mask": 0.0, "policy": 0.0, "step": 0.0}
    steps = 0
    lands = []
    for m in maps:
        s = E.from_grid(m["grid"])
        H, W = s.H, s.W
        trk = [F.Tracker(H, W), F.Tracker(H, W)]
        while not s.done and s.time < max_turns:
            acts = []
            for p in (0, 1):
                t0 = time.perf_counter()
                obs = E.observe(s, p)
                t1 = time.perf_counter()
                planes, scal, (O, A, T) = trk[p].update(obs)
                t2 = time.perf_counter()
                mask = F.legal_mask(O, A, T, H, W, planes)
                t3 = time.perf_counter()
                a = F.index_action(army_weighted_action(mask, A, rng))
                t4 = time.perf_counter()
                tm["observe"] += t1 - t0
                tm["features"] += t2 - t1
                tm["mask"] += t3 - t2
                tm["policy"] += t4 - t3
                acts.append(a)
            t0 = time.perf_counter()
            E.step(s, acts)
            tm["step"] += time.perf_counter() - t0
            steps += 1
        lands.append((s.time, s.land(0), s.land(1)))
    tot = sum(tm.values())
    print(json.dumps({"steps": steps, "steps_per_s": round(steps / tot, 1),
                      "us_per_step": {k: round(v / steps * 1e6, 1) for k, v in tm.items()},
                      "sim_only_steps_per_s": round(steps / (tm["observe"] + tm["step"]), 1),
                      "sim+feat_steps_per_s": round(steps / (tm["observe"] + tm["step"] + tm["features"] + tm["mask"]), 1),
                      "final_turn_land": lands}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=1200)
    a = ap.parse_args()
    run(a.games, a.max_turns)

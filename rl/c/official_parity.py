"""Exact parity: our sim (sim/engine.py observe + step) vs the OFFICIAL evaluator adapter
(rl/c/evaluator/evaluator/engine.py: make_board / jit transition / encode_observation).

For each seed: build the official initial state, mirror it into our sim, let a bot play both
seats from the OFFICIAL observations, step both engines with the same actions and compare
every observation dict (all keys, all cells) and the full board state every turn.

    python rl/c/official_parity.py --seeds 1 2 3 --bot bots/versions/F2.py
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL_DIR = os.environ.get("EVALUATOR_DIR", os.path.join(ROOT, "evaluator"))  # unzip the organizers' evaluator.zip here
sys.path.insert(0, EVAL_DIR)
import engine as OFF  # noqa: E402  (official adapter; puts its vendored engine on sys.path)
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def mirror(snap):
    """Our sim state from the official snapshot at turn 0."""
    H = len(snap["armies"])
    W = len(snap["armies"][0])
    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if not snap["passable"][r][c]:
                grid[r][c] = -2
    gens = snap["generals"]
    for r in range(H):
        for c in range(W):
            if gens[r][c]:
                owner = snap["ownership"]
                # ownership is per player: [p][r][c]
                p = 0 if owner[0][r][c] else 1
                grid[r][c] = 1 + p
    return E.from_grid(grid)


def state_of(s):
    H, W = s.H, s.W
    arm = [[s.army[r * W + c] for c in range(W)] for r in range(H)]
    own = [[[1 if s.owner[r * W + c] == p else 0 for c in range(W)] for r in range(H)] for p in (0, 1)]
    cas = [[1 if s.castle[r * W + c] else 0 for c in range(W)] for r in range(H)]
    return arm, own, cas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--bot", default="bots/versions/F2.py")
    ap.add_argument("--max-turns", type=int, default=1200)
    args = ap.parse_args()
    total_obs = 0
    for seed in args.seeds:
        st = OFF.initial(seed)
        snap = OFF.snapshot(st)
        s = mirror(snap)
        bots = [load(os.path.join(ROOT, args.bot), f"b{seed}_{p}") for p in (0, 1)]
        turn = 0
        while True:
            offo = [OFF.observation(st, p) for p in (0, 1)]
            ours = [E.observe(s, p) for p in (0, 1)]
            for p in (0, 1):
                for k in offo[p]:
                    a, b = offo[p][k], ours[p].get(k)
                    if a != b:
                        if isinstance(a, list):
                            diffs = [(r, c, a[r][c], b[r][c]) for r in range(len(a)) for c in range(len(a[0]))
                                     if a[r][c] != b[r][c]]
                            raise SystemExit(f"seed {seed} turn {turn} p{p} key {k}: {len(diffs)} cells differ, e.g. {diffs[:5]}")
                        raise SystemExit(f"seed {seed} turn {turn} p{p} key {k}: official {a!r} ours {b!r}")
            total_obs += 2
            sn = OFF.snapshot(st)
            arm, own, cas = state_of(s)
            if sn["armies"] != arm or [sn["ownership"][0], sn["ownership"][1]] != own:
                raise SystemExit(f"seed {seed} turn {turn}: board state differs")
            if int(sn["winner"]) >= 0 or s.done or turn >= args.max_turns:
                break
            acts = [list(bots[p].act(offo[p])) for p in (0, 1)]
            st, _ = OFF.step(st, acts)
            E.step(s, acts)
            turn += 1
        print(f"seed {seed}: identical for {turn} turns; winner official={int(OFF.snapshot(st)['winner'])} ours={s.winner}", flush=True)
    print(f"OK: {total_obs} observations identical")


if __name__ == "__main__":
    main()

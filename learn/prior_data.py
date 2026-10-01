"""Generate (candidate features, is_true_general) rows from the map pool.

    python learn/prior_data.py --start 10000 --count 6000 --out data/prior.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

MOD = None


def bot_mod():
    global MOD
    if MOD is None:
        spec = importlib.util.spec_from_file_location("pb", os.path.join(ROOT, "bots", "participant.py"))
        MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MOD)
    return MOD


def one(line):
    m = json.loads(line)
    s = E.from_grid(m["grid"])
    rows = []
    mod = bot_mod()
    for p in (0, 1):
        obs = E.observe(s, p)
        b = mod.Bot(obs)
        T, O, A = b.parse(obs)
        b.turn = 0
        b.general = s.gpos[p]
        b.setup(T)
        true = s.gpos[1 - p]
        grp = f"{m['seed']}_{p}"
        for c in b.cands0:
            rows.append({"g": grp, "y": int(c == true), "f": b.cand_features(c)})
        if true not in b.cands0:
            rows.append({"g": grp, "y": -1, "f": None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=10000)
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--out", default="data/prior.jsonl")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "data", "maps.jsonl")) as f:
        lines = f.readlines()[args.start:args.start + args.count]
    with Pool(args.workers) as pool, open(args.out, "w") as out:
        for rows in pool.imap_unordered(one, lines, chunksize=8):
            for r in rows:
                out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

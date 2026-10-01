"""Build the behaviour-cloning dataset from parsed replays (data/acts.jsonl).

Re-simulates each game exactly with sim/engine.py, tracks the target player's
fogged observations with bc_features.Tracker and samples (planes, scalars,
legal-mask bits, action index) tuples. Output: data/bc/<name>_{planes,scal,mask,act,win}.npy

    python learn/bc_dataset.py --players ResBot --max-games 2500 --sample 0.35 --name resbot
"""
import argparse
import gzip
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "learn"))
from sim import engine as E  # noqa: E402
import bc_features as F  # noqa: E402

RDIR = os.path.join(ROOT, "data", "replays")


def one(args):
    rec, players, sample, seed = args
    rng = random.Random(seed)
    d = json.loads(gzip.open(os.path.join(RDIR, rec["file"])).read())
    H, W = d["dims"]["rows"], d["dims"]["cols"]
    s = E.State(H, W, [r * W + c for r, c in d["mountains"]], [r * W + c for r, c in d["generals"]])
    who = [p for p in (0, 1) if d["players"][p] in players]
    if not who or rec["fails"] > 0:
        return []
    trk = {p: F.Tracker(H, W) for p in who}
    out = []
    for t, acts in enumerate(rec["acts"]):
        if acts is None:
            return out
        for p in who:
            obs = E.observe(s, p)
            planes, scal, (O, A, T) = trk[p].update(obs)
            if rng.random() < sample:
                mask = F.legal_mask(O, A, T, H, W, planes)
                ai = F.action_index(acts[p], W)
                if not mask[ai]:
                    ai = F.PASS_IDX  # invalid move in the replay == pass
                out.append((planes, scal, np.packbits(mask), ai, int(d.get("winner", -1) == p)))
        E.step(s, acts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", nargs="+", default=["ResBot"])
    ap.add_argument("--max-games", type=int, default=2500)
    ap.add_argument("--sample", type=float, default=0.35)
    ap.add_argument("--name", default="resbot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    recs = []
    for line in open(os.path.join(ROOT, "data", "acts.jsonl")):
        r = json.loads(line)
        if any(p in args.players for p in r["players"]) and r["fails"] == 0:
            recs.append(r)
        if len(recs) >= args.max_games:
            break
    print(f"{len(recs)} games", file=sys.stderr)
    od = os.path.join(ROOT, "data", "bc", args.name)
    os.makedirs(od, exist_ok=True)
    buf = {"planes": [], "scal": [], "mask": [], "act": [], "win": []}
    shard = 0
    total = 0

    def flush():
        nonlocal shard, buf
        if not buf["act"]:
            return
        np.save(os.path.join(od, f"planes_{shard:03d}.npy"), np.stack(buf["planes"]))
        np.save(os.path.join(od, f"scal_{shard:03d}.npy"), np.stack(buf["scal"]))
        np.save(os.path.join(od, f"mask_{shard:03d}.npy"), np.stack(buf["mask"]))
        np.save(os.path.join(od, f"act_{shard:03d}.npy"), np.array(buf["act"], np.int32))
        np.save(os.path.join(od, f"win_{shard:03d}.npy"), np.array(buf["win"], np.int8))
        shard += 1
        buf = {"planes": [], "scal": [], "mask": [], "act": [], "win": []}

    with Pool(args.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(
                one, [(r, set(args.players), args.sample, i) for i, r in enumerate(recs)], chunksize=4)):
            for planes, scal, mask, ai, win in rows:
                buf["planes"].append(planes)
                buf["scal"].append(scal)
                buf["mask"].append(mask)
                buf["act"].append(ai)
                buf["win"].append(win)
                total += 1
            if len(buf["act"]) >= 20000:
                flush()
            if (k + 1) % 100 == 0:
                print(f"[bc] {k + 1}/{len(recs)} games, {total} samples", file=sys.stderr)
    flush()
    print(f"saved {total} samples in {shard} shards", file=sys.stderr)


if __name__ == "__main__":
    main()

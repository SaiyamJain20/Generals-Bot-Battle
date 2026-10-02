"""Evaluate Track C checkpoints (deterministic policy, both seats) against an opponent matrix.

    python rl/c/eval_ckpt.py --ckpt rl/c/runs/c1/policy_it99.json [more.json ...] \
        --maps data/maps_fresh.jsonl --offset 1550 --pairs 25 --workers 36 [--opps ...] [--ref bots/versions/F2.py]

--ref also plays the reference heuristic bot file on the same maps (paired comparison).
Prints one JSON line per bot: per-opponent score and pooled score.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import worker as Wk  # noqa: E402

PC = os.path.join(HERE, "participant_c.py")
MATRIX = ["bots/versions/F2.py", "bots/versions/t1c.py", "bots/versions/c_a2es3.py",
          "bots/opp/bc_resbot96.py", "bots/opp/bc_resbot128.py", "bots/opp/bc_nanomena96.py", "bots/opp/bc_kubic96.py",
          "bots/opp/ext_ksolmann.py", "bots/opp/zoo_flash.py", "bots/opp/zoo_mixed.py", "bots/opp/zoo_sniper.py",
          "bots/opp/rusher.py", "bots/opp/hunter.py"]


def _run(t):
    try:
        return Wk.play_c(t)
    except Exception as e:
        return {"outcome": None, "error": str(e), "opp": t["opp"], "label": t.get("label")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--maps", default="data/maps_fresh.jsonl")
    ap.add_argument("--offset", type=int, default=1550)
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--opps", nargs="*", default=None)
    args = ap.parse_args()
    opps = [os.path.join(ROOT, o) for o in (args.opps or MATRIX)]
    opps = [o for o in opps if os.path.exists(o)]
    bots = [(os.path.basename(c), PC, json.load(open(c))) for c in args.ckpt]
    bots += [(os.path.basename(r), os.path.join(ROOT, r) if not os.path.isabs(r) else r, None) for r in args.ref]
    tasks = []
    for label, path, params in bots:
        for o in opps:
            for k in range(2 * args.pairs):
                tasks.append(dict(bot=path, opp=o, map_idx=args.offset + k // 2, seat=k % 2, params=params,
                                  sigma=0, W=8, seed=0, maps=args.maps, label=label))
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with Pool(workers, maxtasksperchild=20) as pool:
        res = pool.map(_run, tasks, chunksize=1)
    for label, _, _ in bots:
        per, allv, errs = {}, [], 0
        for t, r in zip(tasks, res):
            if t["label"] != label:
                continue
            if r.get("outcome") is None:
                errs += 1
                continue
            v = 0.5 * (r["outcome"] + 1.0)
            per.setdefault(os.path.basename(t["opp"]), []).append(v)
            allv.append(v)
        out = {"bot": label, "pooled": round(sum(allv) / max(1, len(allv)), 4), "n": len(allv), "errors": errs,
               **{k: round(sum(v) / len(v), 3) for k, v in per.items()}}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

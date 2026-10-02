"""Collect tuned parameter sets from all runs and build candidate bots on the final base.

    python tools/final_candidates.py --base bots/versions/tune_base8.py --runs runs/ada5c runs/ada6c runs/local7b

For every run directory it looks for mean_avg.json, mean.json and best.json and writes
bots/versions/f_<run>_<kind>.py. Prints the list of created files (feed it to abmulti).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="bots/versions/tune_base8.py")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["mean_avg", "mean", "best"])
    args = ap.parse_args()
    made = []
    for rd in args.runs:
        name = os.path.basename(rd.rstrip("/"))
        for kind in args.kinds:
            f = os.path.join(ROOT, rd, kind + ".json")
            if not os.path.exists(f):
                continue
            params = json.load(open(f))
            out = os.path.join("bots", "versions", f"f_{name}_{kind}.py")
            subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_variant.py"),
                            os.path.join(ROOT, args.base), os.path.join(ROOT, out), json.dumps(params)], check=True)
            made.append(out)
    print(" ".join(made))


if __name__ == "__main__":
    main()

"""Assemble the minimal RL-track tree for Ada in .scratch/stage_rl/code (then rsync it to ~/botbattle-saiyam/rl/code/).

    .venv312/bin/python rl/stage_ada.py [--bc-nets ResBot_96x8.pt ...]
"""
import argparse
import glob
import gzip
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, ".scratch", "stage_rl", "code")


def cp(src, dst_rel):
    dst = os.path.join(OUT, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-nets", nargs="*", default=["ResBot_96x8.pt"])
    args = ap.parse_args()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    files = (glob.glob(os.path.join(ROOT, "sim", "*.py")) + glob.glob(os.path.join(ROOT, "rl", "*.py"))
             + glob.glob(os.path.join(ROOT, "rl", "hier", "*.py")) + glob.glob(os.path.join(ROOT, "rl", "hier", "*.json"))
             + glob.glob(os.path.join(ROOT, "rl", "hier", "*.txt")) + glob.glob(os.path.join(ROOT, "rl", "slurm", "*"))
             + glob.glob(os.path.join(ROOT, "rl", "students", "*.pt"))
             + glob.glob(os.path.join(ROOT, "rl", "students", "ada", "*.pt"))
             + glob.glob(os.path.join(ROOT, "rl", "bots", "*.py"))
             + glob.glob(os.path.join(ROOT, "arena", "*.py")) + [os.path.join(ROOT, "tools", "abmulti.py")])
    for f in files:
        cp(os.path.relpath(f, ROOT), os.path.relpath(f, ROOT))
    for f in ("learn/bc_features.py", "learn/bc_train.py", "bots/versions/tune_base12g.py", "bots/versions/F2.py",
              "bots/versions/t1c.py", "bots/versions/c_a2es3.py"):
        cp(f, f)
    for f in glob.glob(os.path.join(ROOT, "bots", "opp", "*.py")):
        if not os.path.basename(f).startswith("ext_"):
            cp(os.path.relpath(f, ROOT), os.path.relpath(f, ROOT))
    for n in args.bc_nets:
        cp(os.path.join("data", "bc", n), os.path.join("data", "bc", n))
    # strong learned opponent (local-only code, never part of a submission): KSolmann 3M transformer
    ks = os.path.join("vendor", "ext", "KSolmann_generals-bot-training", "agents", "current_standalone")
    for f in ("averagejoe_model.safetensors", "main_plain.py", "main.py"):
        cp(os.path.join(ks, f), os.path.join(ks, f))
    for f in ("bots/opp/ext_stdio.py", "bots/opp/ext_ksolmann.py"):
        cp(f, f)
    # vendor/ext/_pylib (laptop-built safetensors) is NOT staged: the Ada job venv installs safetensors itself
    for name in ("maps.jsonl", "maps_fresh.jsonl"):
        with open(os.path.join(ROOT, "data", name), "rb") as fi, \
                gzip.open(os.path.join(OUT, "data", name + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)
    tot = cnt = 0
    for d, _, fs in os.walk(OUT):
        for f in fs:
            cnt += 1
            tot += os.path.getsize(os.path.join(d, f))
    print(f"staged {cnt} files, {tot / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()

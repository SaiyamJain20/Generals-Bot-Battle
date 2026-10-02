"""Evaluate several bot files against the same pool on the same maps.
python tools/abmulti.py GAMES WORKERS --bots a.py b.py --opps x.py y.py [--offset 600]"""
import math
import os
import sys
sys.path.insert(0, ".")
import arena.run as R

if __name__ == "__main__":
    a = sys.argv[1:]
    n, w = int(a[0]), int(a[1])
    if w <= 0:
        w = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    off = 600
    if "--offset" in a:
        off = int(a[a.index("--offset") + 1])
    bots = a[a.index("--bots") + 1:a.index("--opps")]
    opps = [x for x in a[a.index("--opps") + 1:] if not x.startswith("--") and not x.isdigit()]
    for bot in bots:
        tot = cnt = 0
        per = {}
        for o in opps:
            res = R.run_match(bot, o, n, workers=w, quiet=True, map_offset=off)
            sc = sum(r["a_score"] for r in res)
            tot += sc
            cnt += len(res)
            per[o.split("/")[-1]] = round(sc / len(res), 3)
            fz = sum(1 for r in res if r["reason"].startswith("forfeit") and r["a_score"] == 0)
            if fz:
                per[o.split("/")[-1] + "_OUR_FORFEITS"] = fz
        m = tot / cnt
        print(bot, "pooled", round(m, 4), "+-", round(1.96 * math.sqrt(m * (1 - m) / cnt), 3), per, flush=True)

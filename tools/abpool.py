"""A/B a bot (with PARAMS overrides) against a pool; prints per-opponent and pooled score.
python tools/abpool.py BOT.py GAMES WORKERS 'spec1' 'spec2' ... --opps a.py b.py"""
import json
import math
import sys
sys.path.insert(0, ".")
import arena.run as R

if __name__ == "__main__":
    args = sys.argv[1:]
    k = args.index("--opps")
    bot, n, w = args[0], int(args[1]), int(args[2])
    if w <= 0:
        import os
        w = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    specs, opps = args[3:k] or ["{}"], args[k + 1:]
    for spec in specs:
        params = json.loads(spec)
        tot, cnt, per = 0.0, 0, {}
        for o in opps:
            res = R.run_match(bot, o, n, workers=w, params=(params, None), quiet=True, map_offset=600)
            sc = sum(r["a_score"] for r in res)
            tot += sc
            cnt += len(res)
            per[o.split("/")[-1]] = round(sc / len(res), 3)
        m = tot / cnt
        print(spec, "pooled", round(m, 4), "+-", round(1.96 * math.sqrt(m * (1 - m) / cnt), 3), per, flush=True)

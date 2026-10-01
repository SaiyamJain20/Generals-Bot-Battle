"""Like ab.py but with a configurable worker count (for torch opponents)."""
import json
import sys
sys.path.insert(0, ".")
import arena.run as R

if __name__ == "__main__":
    a, b, n, w = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    for spec in sys.argv[5:] or ["{}"]:
        params = json.loads(spec)
        res = R.run_match(a, b, n, workers=w, params=(params, None), quiet=True, map_offset=300)
        sm = R.summarize(res)
        reasons = {}
        for r in res:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        print(spec, {k: sm[k] for k in ("score", "W", "D", "L", "avg_turns", "forfeits")}, reasons, flush=True)

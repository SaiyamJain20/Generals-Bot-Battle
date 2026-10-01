"""Quick A/B of PARAMS overrides: python tools/ab.py bots/participant.py bots/opp/hunter.py 200 '{"k": v}' ..."""
import json
import sys
sys.path.insert(0, ".")
import arena.run as R

if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for spec in sys.argv[4:] or ["{}"]:
        params = json.loads(spec)
        res = R.run_match(a, b, n, workers=15, params=(params, None), quiet=True)
        sm = R.summarize(res)
        print(spec, {k: sm[k] for k in ("score", "W", "D", "L", "elo", "avg_turns", "forfeits", "A_p99_ms_max", "A_max_ms", "A_first_ms")}, flush=True)

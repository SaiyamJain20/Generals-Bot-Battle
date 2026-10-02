"""In-process match runner for single-file `act(observation)` bots.

Mirrors the Code Bot event contract:
  * a fresh module per game (globals reset), observation is a fresh dict
  * return must be a list/tuple of exactly 5 plain ints (bool is malformed),
    kind in {0,1,2}, dir in 0..3, split in {0,1}; otherwise forfeit
  * any exception -> forfeit; time over limit -> forfeit (when enforced)
Games use exact maps from data/maps.jsonl and sim/engine.py.

    python arena/run.py A.py B.py --games 200 --workers 14 [--limit-ms 150] [--out runs/x.jsonl]

Games are played in side-swapped pairs on the same map.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

_MAPS = None
_COUNTER = [0]


class _LazyMaps:
    """Raw JSON lines kept as strings; each map parsed on first use (saves ~130 MB per worker)."""

    def __init__(self, path):
        with open(path) as f:
            self.lines = f.readlines()
        self.cache = {}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        m = self.cache.get(i)
        if m is None:
            m = json.loads(self.lines[i])
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[i] = m
        return m


def maps():
    global _MAPS
    if _MAPS is None:
        _MAPS = _LazyMaps(os.path.join(ROOT, "data", "maps.jsonl"))
    return _MAPS


def load_bot(path, params=None):
    _COUNTER[0] += 1
    name = f"bot_{os.getpid()}_{_COUNTER[0]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if params and hasattr(mod, "PARAMS"):
        mod.PARAMS.update(params)
    return mod


def validate(a):
    if not isinstance(a, (list, tuple)) or len(a) != 5:
        return False
    for v in a:
        if type(v) is not int or not (-2**31 <= v < 2**31):
            return False
    return a[0] in (0, 1, 2) and 0 <= a[3] <= 3 and a[4] in (0, 1)


def play(bot_paths, map_idx, params=(None, None), limit_ms=None, first_limit_ms=10000,
         record=False, max_turns=E.TRUNCATION, log_features=None):
    """Play one game. bot_paths[p] plays as player p. Returns a result dict."""
    m = maps()[map_idx % len(maps())]
    s = E.from_grid(m["grid"])
    res = {"map": map_idx, "H": m["H"], "W": m["W"], "bots": list(bot_paths),
           "winner": -1, "reason": "draw", "turns": 0, "times": [[], []], "error": None}
    mods = []
    for p in (0, 1):
        try:
            mods.append(load_bot(bot_paths[p], params[p]))
        except Exception:
            res.update(winner=1 - p, reason=f"forfeit_import_p{p}", error=traceback.format_exc())
            return finish(res)
    frames = [] if record else None
    while not s.done and s.time < max_turns:
        acts = []
        for p in (0, 1):
            obs = E.observe(s, p)
            t0 = time.perf_counter()
            try:
                a = mods[p].act(obs)
            except Exception:
                res.update(winner=1 - p, reason=f"forfeit_exception_p{p}", turns=s.time,
                           error=traceback.format_exc())
                return finish(res)
            dt = (time.perf_counter() - t0) * 1000.0
            res["times"][p].append(dt)
            lim = first_limit_ms if s.time == 0 else limit_ms
            if lim is not None and dt > lim:
                res.update(winner=1 - p, reason=f"forfeit_timeout_p{p}", turns=s.time,
                           error=f"{dt:.1f}ms at turn {s.time}")
                return finish(res)
            if not validate(a):
                res.update(winner=1 - p, reason=f"forfeit_malformed_p{p}", turns=s.time,
                           error=repr(a))
                return finish(res)
            acts.append(list(a))
        if record:
            frames.append({"t": s.time, "army": s.army[:], "owner": s.owner[:],
                           "castle": [i for i, c in enumerate(s.castle) if c], "acts": acts})
        E.step(s, acts)
    res["turns"] = s.time
    res["winner"] = s.winner
    if s.winner >= 0:
        res["reason"] = "win_late" if s.time > E.DEATHTOUCH_TURN else "win"
    res["final"] = {"land": [s.land(0), s.land(1)], "army": [s.total_army(0), s.total_army(1)],
                    "castles": [sum(1 for i, c in enumerate(s.castle) if c and s.owner[i] == p) for p in (0, 1)]}
    if record:
        res["frames"] = frames
        res["mountains"] = [i for i, v in enumerate(s.mountain) if v]
        res["gpos"] = s.gpos
    return finish(res)


def finish(res):
    for p in (0, 1):
        ts = sorted(res["times"][p])
        if ts:
            rest = sorted(res["times"][p][1:]) or [0.0]
            res.setdefault("tstats", []).append({
                "first": res["times"][p][0], "max": rest[-1],
                "p99": rest[min(len(rest) - 1, int(0.99 * len(rest)))], "mean": sum(ts) / len(ts)})
        else:
            res.setdefault("tstats", []).append(None)
    del res["times"]
    return res


def _task(args):
    a, b, map_idx, swap, kw = args
    paths = (b, a) if swap else (a, b)
    params = kw.pop("params", (None, None))
    if swap:
        params = (params[1], params[0])
    r = play(paths, map_idx, params=params, **kw)
    r["swap"] = swap
    # score from A's perspective
    a_player = 1 if swap else 0
    r["a_score"] = 0.5 if r["winner"] < 0 else (1.0 if r["winner"] == a_player else 0.0)
    r["a_player"] = a_player
    return r


def summarize(results, label="A"):
    n = len(results)
    if not n:
        return {}
    w = sum(1 for r in results if r["a_score"] == 1.0)
    d = sum(1 for r in results if r["a_score"] == 0.5)
    l = n - w - d
    score = (w + 0.5 * d) / n
    var = sum((r["a_score"] - score) ** 2 for r in results) / max(1, n - 1)
    se = math.sqrt(var / n) if n > 1 else 0.5

    def elo(x):
        x = min(max(x, 1e-3), 1 - 1e-3)
        return -400 * math.log10(1 / x - 1)

    forfeits = {}
    for r in results:
        if r["reason"].startswith("forfeit"):
            forfeits[r["reason"]] = forfeits.get(r["reason"], 0) + 1
    tmax = max((t["max"] for r in results for t in r["tstats"] if t), default=0)
    a_p99 = [r["tstats"][r["a_player"]]["p99"] for r in results if r["tstats"][r["a_player"]]]
    a_max = [r["tstats"][r["a_player"]]["max"] for r in results if r["tstats"][r["a_player"]]]
    a_first = [r["tstats"][r["a_player"]]["first"] for r in results if r["tstats"][r["a_player"]]]
    return {"n": n, "W": w, "D": d, "L": l, "score": round(score, 4),
            "elo": round(elo(score), 1), "elo_lo": round(elo(score - 1.96 * se), 1),
            "elo_hi": round(elo(score + 1.96 * se), 1),
            "avg_turns": round(sum(r["turns"] for r in results) / n, 1),
            "forfeits": forfeits, "A_p99_ms_max": round(max(a_p99, default=0), 2),
            "A_max_ms": round(max(a_max, default=0), 2), "A_first_ms": round(max(a_first, default=0), 1),
            "any_max_ms": round(tmax, 2)}


def run_match(a, b, games, workers=8, map_offset=0, limit_ms=None, params=(None, None),
              out=None, quiet=False):
    tasks = []
    for g in range(games):
        pair, swap = divmod(g, 2)
        tasks.append((a, b, map_offset + pair, bool(swap),
                      {"limit_ms": limit_ms, "params": params}))
    results = []
    with Pool(workers, maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(_task, tasks):
            results.append(r)
            if out:
                with open(out, "a") as f:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "frames"}) + "\n")
            if r["reason"].startswith("forfeit") and not quiet:
                print(f"[arena] {r['reason']} map={r['map']} turn={r['turns']}: "
                      f"{(r['error'] or '')[-600:]}", file=sys.stderr)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--map-offset", type=int, default=0)
    ap.add_argument("--limit-ms", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    res = run_match(args.a, args.b, args.games, args.workers, args.map_offset, args.limit_ms,
                    out=args.out)
    summ = summarize(res)
    summ["wall_s"] = round(time.time() - t0, 1)
    reasons = {}
    for r in res:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summ["reasons"] = reasons
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

"""Track C rollout worker: one full game on the exact simulator with the learner's bot in one seat.

The learner's bot is rl/c/participant_c.py with policy params; when `sigma > 0` its RL hook explores
(Gaussian logits held for W turns) and returns the per-window trace for the policy gradient.
"""
import json
import math
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402
import arena.run as R  # noqa: E402


_MAPCACHE = {}


def get_map(path, idx):
    path = path or os.path.join(ROOT, "data", "maps.jsonl")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    ml = _MAPCACHE.get(path)
    if ml is None:
        ml = _MAPCACHE[path] = R._LazyMaps(path)
    return ml[idx % len(ml)]


def play_c(task):
    """task: dict(bot, opp, map_idx, seat, params, opp_params, sigma, W, seed, maps, max_turns, record_actions)
    Returns dict(outcome, reward parts, trace, turns, reason)."""
    m = get_map(task.get("maps"), task["map_idx"])
    s = E.from_grid(m["grid"])
    seat = task["seat"]
    mods = [None, None]
    out = {"map": task["map_idx"], "seat": seat, "opp": task["opp"], "trace": [], "reason": "draw",
           "error": None, "actions": [] if task.get("record_actions") else None}
    try:
        me = R.load_bot(task["bot"], task.get("params"))
        if task.get("sigma", 0) > 0:
            me.RL = {"rng": random.Random(task["seed"]), "sigma": float(task["sigma"]), "W": int(task["W"]),
                     "trace": []}
        op = R.load_bot(task["opp"], task.get("opp_params"))
    except Exception as e:  # import failure: drop this game
        out["error"] = f"import: {type(e).__name__}: {e}"
        return out
    mods[seat], mods[1 - seat] = me, op
    max_turns = task.get("max_turns", E.TRUNCATION)
    last = None
    while not s.done and s.time < max_turns:
        acts = []
        for p in (0, 1):
            obs = E.observe(s, p)
            try:
                a = mods[p].act(obs)
            except Exception as e:
                out["error"] = f"exception p{p}: {type(e).__name__}: {e}"
                a = None
            if a is None or not R.validate(a):
                # forfeit rules: the offender loses
                out["reason"] = f"forfeit_p{p}"
                out["outcome"] = -1.0 if p == seat else 1.0
                out["turns"] = s.time
                return _fin(out, me, last, R)
            acts.append(list(a))
        if out["actions"] is not None:
            out["actions"].append(acts[seat])
        last = (s.total_army(seat), s.total_army(1 - seat), s.land(seat), s.land(1 - seat))
        E.step(s, acts)
    out["turns"] = s.time
    if s.winner < 0:
        out["outcome"] = 0.0
    else:
        out["outcome"] = 1.0 if s.winner == seat else -1.0
        out["reason"] = "win" if s.winner == seat else "loss"
    return _fin(out, me, last, R)


def _fin(out, me, last, R):
    rl = getattr(me, "RL", None)
    if rl is not None:
        out["trace"] = [{"t": w["t"], "f": w["f"], "mu": w["mu"], "a": w["a"]} for w in rl["trace"]]
    if last:
        my, op, ml, ol = last
        out["army_ratio"] = math.log(max(1, my) / max(1, op))
        out["land_ratio"] = math.log(max(1, ml) / max(1, ol))
    else:
        out["army_ratio"] = out["land_ratio"] = 0.0
    try:
        import gc
        gc.unfreeze()
        gc.collect()
    except Exception:
        pass
    return out

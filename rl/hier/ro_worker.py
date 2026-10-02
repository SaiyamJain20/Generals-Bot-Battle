"""Rollout worker for RO-PPO (no torch import: workers stay small).

play_pair(task) plays one map from BOTH seats against one opponent and returns the recorded
decisions of the RO policy.  bot_ro.py is loaded fresh per game, so module globals never leak.
"""
import gc
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, os.path.join(ROOT, "arena")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim import engine as E  # noqa: E402
import run as AR  # noqa: E402  (arena/run.py)

BOT_RO = os.path.join(HERE, os.environ.get("RO_BOT", "bot_ro.py"))  # RO_BOT=bot_ro_F2.py for the F2 base


def load_ro(weights, sample=False, record=None, seed=0):
    mod = AR.load_bot(BOT_RO)
    mod.RO_W = weights
    mod.RO_SAMPLE = bool(sample)
    mod.RO_RECORD = record
    mod.RO_RNG = random.Random(seed)
    return mod


def load_opp(spec, seed=0):
    """spec: {"name","path"} or {"name","snap": weights}"""
    if "snap" in spec:
        return load_ro(spec["snap"], sample=True, record=None, seed=seed)
    return AR.load_bot(spec["path"])


def play_game(ro_seat, map_idx, weights, opp_spec, sample=True, seed=0, record=True, max_turns=None):
    """One game; RO policy sits in seat `ro_seat`. Returns dict."""
    m = AR.maps()[map_idx % len(AR.maps())]
    s = E.from_grid(m["grid"])
    recs = [] if record else None
    res = {"map": map_idx, "seat": ro_seat, "opp": opp_spec["name"], "err": None}
    try:
        ro = load_ro(weights, sample=sample, record=recs, seed=seed)
        op = load_opp(opp_spec, seed=seed + 1)
    except Exception as e:  # unloadable opponent
        res.update(outcome=None, recs=[], turns=0, err="load:" + repr(e)[:200], ro_ms=0.0)
        return res
    mods = [None, None]
    mods[ro_seat] = ro
    mods[1 - ro_seat] = op
    limit = E.TRUNCATION if max_turns is None else max_turns
    ro_t = 0.0
    t_wall = time.perf_counter()
    winner_forced = None
    while not s.done and s.time < limit:
        acts = []
        for p in (0, 1):
            obs = E.observe(s, p)
            t0 = time.perf_counter()
            try:
                a = mods[p].act(obs)
            except Exception as e:
                res["err"] = "exc_p%d:%s" % (p, repr(e)[:150])
                winner_forced = 1 - p
                break
            if p == ro_seat:
                ro_t += time.perf_counter() - t0
            if not AR.validate(a):
                res["err"] = "malformed_p%d" % p
                winner_forced = 1 - p
                break
            acts.append(list(a))
        if winner_forced is not None:
            break
        E.step(s, acts)
    winner = winner_forced if winner_forced is not None else (s.winner if s.done else -1)
    if winner < 0:
        out = 0
    else:
        out = 1 if winner == ro_seat else -1
    res.update(outcome=out, recs=recs, turns=s.time, wall=time.perf_counter() - t_wall, ro_ms=1000.0 * ro_t / max(1, s.time),
               draw=(winner < 0), forfeit=winner_forced is not None)
    # bots call gc.freeze(); undo it so finished games are reclaimed
    del mods, ro, op
    gc.unfreeze()
    gc.collect()
    return res


def play_pair(task):
    """task: map_idx, weights, opp (spec), seed, sample, record, max_turns."""
    out = []
    for seat in (0, 1):
        out.append(play_game(seat, task["map_idx"], task["weights"], task["opp"],
                             sample=task.get("sample", True), seed=task["seed"] * 2 + seat,
                             record=task.get("record", True), max_turns=task.get("max_turns")))
    return out


def init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")

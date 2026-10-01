"""The submission must always return 5 plain ints and never raise."""
import importlib.util
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from sim import engine as E  # noqa: E402

BOT = os.path.join(ROOT, "bots", "participant.py")


def fresh():
    spec = importlib.util.spec_from_file_location("pb_%d" % random.randrange(10 ** 9), BOT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def valid(a):
    return (isinstance(a, list) and len(a) == 5 and all(type(v) is int for v in a)
            and a[0] in (0, 1, 2) and 0 <= a[3] <= 3 and a[4] in (0, 1))


def test_garbage_observations():
    m = fresh()
    rng = random.Random(0)
    for k in range(300):
        H, W = rng.randint(18, 21), rng.randint(18, 21)
        obs = {"turn": rng.randint(0, 1199), "height": H, "width": W, "player_id": rng.randint(0, 1),
               "my_land": rng.randint(0, 300), "my_army": rng.randint(0, 3000),
               "opp_land": rng.randint(0, 300), "opp_army": rng.randint(0, 3000),
               "type": [[rng.choice([0, 1, 2, 3, 4, 5]) for _ in range(W)] for _ in range(H)],
               "owner": [[rng.choice([0, 1, 2]) for _ in range(W)] for _ in range(H)],
               "army": [[rng.randint(0, 50) for _ in range(W)] for _ in range(H)]}
        if k % 3 == 1:
            obs = {kk: (np.array(v) if isinstance(v, list) else v) for kk, v in obs.items()}
        if k % 3 == 2:
            for g in ("type", "owner", "army"):
                obs[g] = [v for row in obs[g] for v in row]
        assert valid(m.act(obs))
    assert valid(m.act({}))
    assert valid(m.act(None))


def test_all_sizes_full_games():
    maps = [json.loads(l) for _, l in zip(range(400), open(os.path.join(ROOT, "data", "maps.jsonl")))]
    seen = {}
    for mp in maps:
        seen.setdefault((mp["H"], mp["W"]), mp)
    assert len(seen) == 16
    for (H, W), mp in sorted(seen.items()):
        s = E.from_grid(mp["grid"])
        bots = [fresh(), fresh()]
        while not s.done and s.time < 400:
            acts = []
            for p in (0, 1):
                a = bots[p].act(E.observe(s, p))
                assert valid(a), (H, W, s.time, a)
                acts.append(a)
            E.step(s, acts)
        assert bots[0]._BOT.err == 0 and bots[1]._BOT.err == 0, (H, W)


def test_numpy_observation_game():
    mp = json.loads(open(os.path.join(ROOT, "data", "maps.jsonl")).readline())
    s = E.from_grid(mp["grid"])
    m = fresh()
    while not s.done and s.time < 200:
        o = E.observe(s, 0)
        o = {k: (np.array(v, dtype=np.int32) if isinstance(v, list) else np.int64(v)) for k, v in o.items()}
        a = m.act(o)
        assert valid(a)
        E.step(s, [a, [1, 0, 0, 0, 0]])
    assert m._BOT.err == 0

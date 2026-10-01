"""Trace a game between two bots, printing periodic stats. Debug helper."""
import importlib.util
import json
import sys
from collections import Counter

sys.path.insert(0, ".")
from sim import engine as E


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "DEBUG"):
        m.DEBUG = True
    return m


def main(a, b, map_idx=0, every=50, until=1200):
    maps = [json.loads(l) for i, l in zip(range(map_idx + 1), open("data/maps.jsonl"))]
    s = E.from_grid(maps[map_idx]["grid"])
    bots = [load(a, "A"), load(b, "B")]
    kinds = [Counter(), Counter()]
    while not s.done and s.time < until:
        acts = []
        for p in (0, 1):
            x = bots[p].act(E.observe(s, p))
            kinds[p][x[0]] += 1
            acts.append(x)
        E.step(s, acts)
        if s.time % every == 0:
            st = [(s.land(p), s.total_army(p), s.army[s.gpos[p]],
                   sum(1 for i, c in enumerate(s.castle) if c and s.owner[i] == p)) for p in (0, 1)]
            print(s.time, "land/army/gen/castles", st, dict(kinds[0]), dict(kinds[1]))
            kinds = [Counter(), Counter()]
    print("end", s.time, "winner", s.winner)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)

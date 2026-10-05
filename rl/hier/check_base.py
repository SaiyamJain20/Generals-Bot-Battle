"""Check that an RO bot with RO_W=None plays move-for-move like its base heuristic, and that reference
weights (u=v=0) pick the heuristic's top option on every decision.

    python rl/hier/check_base.py rl/hier/bot_ro_F2.py bots/versions/F2.py [games]
"""
import importlib.util
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rl", "hier"))
from sim import engine as E  # noqa: E402


def load(path, tag):
    spec = importlib.util.spec_from_file_location("chk_%s_%d" % (tag, random.randrange(10 ** 9)), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "PARAMS"):
        m.PARAMS["soft_budget_ms"] = 10 ** 6   # no time-budget truncation, so the check is deterministic under load
        m.PARAMS["first_budget_ms"] = 10 ** 6
    return m


def play(a_path, maps_idx, ro_w=None, record=None, max_turns=300):
    lines = open(os.path.join(ROOT, "data", "maps.jsonl")).readlines()
    s = E.from_grid(json.loads(lines[maps_idx])["grid"])
    a = load(a_path, "a")
    if ro_w is not None:
        a.RO_W = ro_w
        a.RO_RECORD = record
    b = load(os.path.join(ROOT, "bots", "opp", "zoo_mixed.py"), "b")
    moves = []
    while not s.done and s.time < max_turns:
        x = a.act(E.observe(s, 0))
        y = b.act(E.observe(s, 1))
        moves.append(tuple(int(v) for v in x))
        E.step(s, [x, y])
    return moves


def main():
    ro_path, base_path = sys.argv[1], sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    import ro_ppo as R
    ok = True
    for g in range(games):
        m_base = play(base_path, 15000 + g)
        m_none = play(ro_path, 15000 + g)
        same = m_base == m_none
        rec = []
        m_ref = play(ro_path, 15000 + g, ro_w=R.default_weights(hidden=16), record=rec)
        same_ref = m_base == m_ref
        print(f"game {g}: RO_W=None identical={same} ({len(m_base)} turns); ref weights identical={same_ref} "
              f"decisions={len(rec)}")
        ok = ok and same and same_ref
    print("ALL OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

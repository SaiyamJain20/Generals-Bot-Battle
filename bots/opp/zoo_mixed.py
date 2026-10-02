"""zoo_mixed: picks one zoo style per game, seeded from the map (own general cell + board size)."""
import importlib.util
import os

PASS = [1, 0, 0, 0, 0]
STYLES = ["flash", "castler", "gatherer", "sniper", "turtle_dt", "expander_plus"]
_ACT = None


def _load(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zoo_%s.py" % name)
    spec = importlib.util.spec_from_file_location("zoo_mixed_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.act


def act(observation):
    global _ACT
    try:
        if _ACT is None or int(observation["turn"]) == 0:
            H, W = int(observation["height"]), int(observation["width"])
            seed = H * 1009 + W * 31
            for r, row in enumerate(observation["type"]):
                for c, v in enumerate(row):
                    if v == 4 and observation["owner"][r][c] == 1:
                        seed += (r * W + c) * 7919
            _ACT = _load(STYLES[seed % len(STYLES)])
        return _ACT(observation)
    except Exception:
        return list(PASS)

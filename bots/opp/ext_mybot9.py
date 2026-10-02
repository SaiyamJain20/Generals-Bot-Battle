"""External sparring partner: "my_bot9" (mrinmoy2developer, upstream generals-bots PR 138), stdlib Python
heuristic: general-funded castle expander, remembered enemy general, scouting behind seen enemy land,
gather for the general strike from turn 500, nearest-stack deathtouch routing.

Source: https://github.com/mrinmoy2developer/gio-competition-tooling @ f624c741ad5084be63fc17bacd3961599b8dcc82
        competition/agents/my_bot9/agent.py (MIT licence in repo).
PARAMS["fix_price"] (default True) sets its castle PROXIMITY_PENALTY 10 -> 14 (the real rule; the original
was written for an older price and issues many unaffordable builds, which our engine turns into passes).
Set False to play the unchanged original. Runs in-process via ext_starterkit.py. LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

PARAMS = {"fix_price": True}
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_s = _u.spec_from_file_location("_ext_sk_mybot9", _os.path.join(_HERE, "ext_starterkit.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
_PATH = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "mybot", "my_bot9", "agent.py"))
_inner = []


def act(obs):
    if not _inner:
        a = _m.make_act(_PATH)
        mod = a.module
        if PARAMS["fix_price"]:
            mod.PROXIMITY_PENALTY = 14
            mod.HOLD_CEILING = mod.BASE_COST + mod.PROXIMITY_PENALTY + mod.BUILD_RESERVE + 1
        _inner.append(a)
    return _inner[0](obs)

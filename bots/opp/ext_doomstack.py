"""External sparring partner: Klincent "doomstack_rusher" (default STRESS_MODE=doomstack).
Expands, collects owned-route donors into a persistent rally, deploys one concentrated attacker.
No castle building, no explicit general defence / deathtouch policy.

Source: https://github.com/Klincent/generals-bots @ 2260b6f19d51a14d7c68770677f22d04dfd88022
        competition/agents/doomstack_rusher/agent.py (MIT licence in repo). Stdlib Python.
Runs the original Agent in-process via ext_starterkit.py. LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_s = _u.spec_from_file_location("_ext_sk_doomstack", _os.path.join(_HERE, "ext_starterkit.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act(_os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "juraj", "doomstack", "agent.py")),
                  env={"STRESS_MODE": None})

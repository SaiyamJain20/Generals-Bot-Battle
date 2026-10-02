"""External sparring partner: relh's extended starter Expander (competition/agents/expander_python):
castle_opening, siege (persistent spearhead to the remembered enemy general), late exploration,
pressure from turn 800. Stdlib Python.

Source: https://github.com/relh/generals-bots @ 04f81887b186ff5a4343a58b52d32a23746d82ae
        competition/agents/expander_python/agent.py (MIT). In-process via ext_starterkit.py.
LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_s = _u.spec_from_file_location("_ext_sk_relhexp", _os.path.join(_HERE, "ext_starterkit.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act(_os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "relh_generals-bots", "competition",
                                                  "agents", "expander_python", "agent.py")))

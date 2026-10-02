"""External sparring partner: bca's "Boss" heuristic (blake-ar), the NumPy competition export of
generals.agents.BossAgent: tactical overrides, shortest-path routing, defensive screening, castle economy
(max 4 castles, spacing 5, until turn 680), fog scouting, deathtouch behaviour. Used by bca as a
training/eval league opponent.

Source: https://github.com/blake-ar/generals-bots @ df93b034f4da2ade4e90234ef83cc9afceb7423d
        competition/agents/boss/agent.py (MIT engine-fork licence). In-process via ext_starterkit.py.
LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
try:
    from threadpoolctl import threadpool_limits as _tl
    _tl(1)
except Exception:
    pass

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_s = _u.spec_from_file_location("_ext_sk_boss", _os.path.join(_HERE, "ext_starterkit.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act(_os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "blake-ar_generals-bots", "competition",
                                                  "agents", "boss", "agent.py")))

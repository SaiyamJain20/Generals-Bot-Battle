"""External sparring partner: Juraj V3.5 deterministic C++17 castle-planning heuristic.
Source: https://github.com/Klincent/generals-bots @ e50123cee7d924f0d643acd372a5300971f93917
        competition/agents/juraj_v35_cpp (MIT licence at that commit).
Runs the original compiled agent as a stdio subprocess via ext_stdio.py
(build once: bash vendor/ext/juraj/juraj_v35/build.sh). LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DIR = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "juraj", "juraj_v35"))
_s = _u.spec_from_file_location("_ext_stdio_juraj35", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_os.path.join(_DIR, "agent")], cwd=_DIR)

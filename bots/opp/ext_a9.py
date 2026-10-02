"""External sparring partner: "A9" trained C++ submission (Marathon entrant mortid0), compiled locally.
Source: https://github.com/mortid0/generals-bots @ 78128d6db6cc41ab2a94ee855349be42589fcc8f (branch Marathon),
        competition/submissions/a9_20260812/atlas-negative-candidate.zip (sha256 matches the repo's SHA256SUMS).
Runs the compiled agent (model.hpp + weights.bin) as a stdio subprocess via ext_stdio.py
(build once: bash vendor/ext/mortid0_a9/my_bot_cpp/build.sh). LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DIR = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "mortid0_a9", "my_bot_cpp"))
_s = _u.spec_from_file_location("_ext_stdio_a9", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_os.path.join(_DIR, "agent")], cwd=_DIR)

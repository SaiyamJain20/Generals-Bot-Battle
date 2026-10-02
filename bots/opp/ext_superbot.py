"""External sparring partner: Juraj final-superbot C++17 (author-published submission ZIP; J35 lineage
plus rear-army collection, muster, doomguard, live castle pricing, anti-cycle).
Source: https://github.com/Klincent/generals-bots @ 704ce13d891af22361cb76b9389da92a36b9903c
        submissions/final-superbot.zip (MIT licence in repo).
Runs the original compiled agent as a stdio subprocess via ext_stdio.py
(build once: bash vendor/ext/juraj/superbot/build.sh). LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_DIR = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "juraj", "superbot"))
_s = _u.spec_from_file_location("_ext_stdio_superbot", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_os.path.join(_DIR, "agent")], cwd=_DIR)

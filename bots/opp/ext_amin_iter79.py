"""Amin PPO CNN, checkpoint main7_iter79 (see ext_amin.py for provenance). LOCAL TESTING ONLY."""
import importlib.util as _u
import os as _os

_s = _u.spec_from_file_location("_ext_amin_base_iter79", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ext_amin.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
_m.PARAMS["ckpt"] = "main7_iter79"
PARAMS = _m.PARAMS
act = _m.act

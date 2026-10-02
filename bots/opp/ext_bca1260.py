"""bca earlier checkpoint: competition/agents/smoke_1260_baseline (smoke_8xh100 EMA iter 1260, legacy_38 obs,
7-layer transformer, JAX CPU). See ext_bca.py for provenance. LOCAL TESTING ONLY. 1 worker."""
import importlib.util as _u
import os as _os

_s = _u.spec_from_file_location("_ext_bca_base_1260", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ext_bca.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
_m.PARAMS["model"] = "smoke_1260_baseline"
PARAMS = _m.PARAMS
act = _m.act

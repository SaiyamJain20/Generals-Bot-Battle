"""External sparring partner: RonitNath/generals-bots built-in 'graph' (GraphSearchAgent)
heuristic (MIT, old-ruleset JAX fork; never builds castles). Run via
vendor/ext/RonitNath_generals-bots/stdio_shim.py as a stdio subprocess. LOCAL TESTING ONLY.
Set AGENT below to another built-in name (material, surround, scout, ...) for siblings."""
import importlib.util as _u
import os as _os

AGENT = "graph"
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "RonitNath_generals-bots"))
_PY = _os.path.normpath(_os.path.join(_HERE, "..", "..", ".venv312", "bin", "python"))
_s = _u.spec_from_file_location("_ext_stdio_ronit_" + AGENT, _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_PY, "-u", "stdio_shim.py"], cwd=_ROOT, env={
    "PYTHONPATH": _ROOT, "AGENT": AGENT, "JAX_PLATFORMS": "cpu", "PYGAME_HIDE_SUPPORT_PROMPT": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})

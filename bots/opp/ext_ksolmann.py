"""External sparring partner: KSolmann/generals-bot-training frozen reference opponent
`opponents/3m_castle_28k` (AverageJoe-V2 history transformer, 3.0M params, trained for the
generals.bot competition rules with castles). Weights exported to safetensors with the repo's
scripts/export_checkpoint.py (needs equinox, not vendored), then run by the repo's own
standalone NumPy/torch runtime (agents/current_standalone/main.py) as a stdio subprocess.
Source: github.com/KSolmann/generals-bot-training (no top-level licence; bundled
generals-bots code is MIT). LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_V = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext"))
_ROOT = _os.path.join(_V, "KSolmann_generals-bot-training", "agents", "current_standalone")
_PY = _os.path.normpath(_os.path.join(_HERE, "..", "..", ".venv312", "bin", "python"))
_s = _u.spec_from_file_location("_ext_stdio_ksol", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_PY, "-u", "main_plain.py"], cwd=_ROOT, env={
    "PYTHONPATH": _os.path.join(_V, "_pylib"),
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

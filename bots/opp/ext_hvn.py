"""External sparring partner: hv-nguyeen heuristic Controller (NumPy) with configs/v18.json
(byte-identical to v9/v13..v17 configs; ladder versions differed by code, only HEAD code is public).
Ladder notes in the repo (docs/ml-log.md): v13 heuristic ~1737 ladder Elo (best heuristic build),
v18 ~1587, their neural bot 1849 (rank 21/86; weights NOT in the repo, so the heuristic plays).

Source: https://github.com/hv-nguyeen/Generals-RL-bot @ cc259256a787a11cb2e0fea04590f604c351f950
        bot/ (main.py stdio loop, policy/controller.py). No licence file in the repo.
Runs the original stdio entry point (python -m bot.main, BOT_CONFIG=configs/v18.json) as a
subprocess via ext_stdio.py. The controller searches until a 110 ms deadline when needed.
LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "hv-nguyeen_Generals-RL-bot"))
_PY = _os.path.normpath(_os.path.join(_HERE, "..", "..", ".venv312", "bin", "python"))
_s = _u.spec_from_file_location("_ext_stdio_hvn", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_PY, "-u", "-m", "bot.main"], cwd=_ROOT, env={
    "PYTHONPATH": _ROOT, "BOT_CONFIG": _os.path.join(_ROOT, "configs", "v18.json"),
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

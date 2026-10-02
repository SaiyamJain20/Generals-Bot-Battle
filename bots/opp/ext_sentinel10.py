"""External sparring partner: relh Sentinel heuristic, SENTINEL_VARIANT=v10, competition mode
(castle building on, deathtouch 800, cap 1200). Hand-written JAX policy: routes surplus to a shared
objective, castle build/capture, general-safety checks. v2 is the author's default (never displaced);
v10 = v9 remembered-general + home mobilisation after deathtouch (best later screen, 4-2).

Source: https://github.com/relh/generals-bots @ 04f81887b186ff5a4343a58b52d32a23746d82ae (MIT)
        competition/agents/sentinel_python/main.py, generals/agents/sentinel*_agent.py
Runs the original stdio entry point as a subprocess (its `generals` package would shadow the pinned
engine, so it must not be imported in-process) via ext_stdio.py. JIT cache is pre-warmed for all 16
shapes under sentinel_python/.jax_cache (warm_cache.py); a cold shape compiles in ~3 s.
~380 MB RSS per process. LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "relh_generals-bots"))
_DIR = _os.path.join(_ROOT, "competition", "agents", "sentinel_python")
_PY = _os.path.normpath(_os.path.join(_HERE, "..", "..", ".venv312", "bin", "python"))
_s = _u.spec_from_file_location("_ext_stdio_sentinel10", _os.path.join(_HERE, "ext_stdio.py"))
_m = _u.module_from_spec(_s)
_s.loader.exec_module(_m)
act = _m.make_act([_PY, "-u", "main.py"], cwd=_DIR, env={
    "PYTHONPATH": _ROOT, "SENTINEL_MODE": "competition", "SENTINEL_VARIANT": "v10",
    "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "JAX_COMPILATION_CACHE_DIR": _os.path.join(_DIR, ".jax_cache"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"})

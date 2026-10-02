"""External sparring partner: "bca" (generals.bot Marathon 2026-09-01 rank 5, 42.6%),
transformer+conv PPO policy, final iteration-1313 EMA weights (bf16), JAX on CPU.

Source: https://github.com/blake-ar/generals-bots @ df93b034f4da2ade4e90234ef83cc9afceb7423d
        competition/agents/conv_1313 (bot.py, weights.npz 25 MB)
Licence: MIT (engine fork; bot code has no separate notice). LOCAL TESTING ONLY.

Runs the original Conv1313Agent in-process (loaded by file path; its module name
`bot` clashes with other repos). The jitted forward is cached process-wide so
only the first game pays the compile. Greedy (argmax) like the original default.
Heavy: ~1 GB RSS, run with ONE worker.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
                                   "inter_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "JAX_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "ext",
                    "blake-ar_generals-bots", "competition", "agents", "conv_1313")
_CACHE = sys.__dict__.setdefault("_porter_bca_cache", {})


os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_DIR, ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")


def _module():
    if "mod" not in _CACHE:
        import jax
        try:  # persistent compile cache keeps the first move well under the 10 s limit
            jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        except Exception:
            pass
        spec = importlib.util.spec_from_file_location("_bca_conv1313_bot", os.path.join(_DIR, "bot.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["mod"] = mod
    return _CACHE["mod"]


_agent = [None, -1]


def _new_agent(H, W):
    mod = _module()
    a = mod.Conv1313Agent(H, W)
    if "forward" in _CACHE:  # reuse compiled function + weights from an earlier game
        a._forward = _CACHE["forward"]
        a.parameters = _CACHE["params"]
    else:
        a.warmup()
        _CACHE["forward"] = a._forward
        _CACHE["params"] = a.parameters
    return a


def act(obs):
    t = obs["turn"]
    if _agent[0] is None or t <= _agent[1]:
        _agent[0] = _new_agent(obs["height"], obs["width"])
    _agent[1] = t
    o = SimpleNamespace(height=obs["height"], width=obs["width"], turn=t, my_land=obs["my_land"],
                        my_army=obs["my_army"], opponent_land=obs["opp_land"],
                        opponent_army=obs["opp_army"], types=obs["type"], owners=obs["owner"],
                        armies=obs["army"])
    return [int(v) for v in _agent[0].act(o)]

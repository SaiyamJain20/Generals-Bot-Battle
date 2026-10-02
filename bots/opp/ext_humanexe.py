"""External sparring partner: bca's JAX "Human.exe" policy, a competition-rules adaptation of
EklipZ's Human.exe (github.com/EklipZgit/generals-bot, MIT): persistent fog beliefs, enemy-general
prediction from emerging territory, tactical overrides, gather/launch cycles, information-aware
expansion, castle build-site economics, urgent general hunt before deathtouch.

Source: https://github.com/blake-ar/generals-bots @ df93b034f4da2ade4e90234ef83cc9afceb7423d
        generals/agents/human_exe_agent.py (+ heuristic_utils.py), MIT. Used there as a league opponent.
PARAMS["agent"] may name another of that repo's JAX heuristics (boss, castle_economist,
deathtouch_clock, draw_grinder, fog_scout, raider); only human_exe carries match memory.

Two modes in one file:
  * imported by the arena: spawns `python ext_humanexe.py <agent>` (this file) as a stdio
    subprocess through ext_stdio.py, because blake-ar's `generals` package would shadow the
    pinned engine in-process;
  * run as a script: the stdio server (handshake, frames, one reply per frame), importing
    blake-ar's package from PYTHONPATH. Persistent JAX compile cache under
    vendor/ext/blake-ar_generals-bots/.jax_cache_porter.
LOCAL TESTING ONLY.
"""
import importlib.util as _u
import os as _os
import sys as _sys

PARAMS = {"agent": "human_exe"}
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.normpath(_os.path.join(_HERE, "..", "..", "vendor", "ext", "blake-ar_generals-bots"))
_PY = _os.path.normpath(_os.path.join(_HERE, "..", "..", ".venv312", "bin", "python"))
_ENV = {"PYTHONPATH": _ROOT, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "JAX_COMPILATION_CACHE_DIR": _os.path.join(_ROOT, ".jax_cache_porter"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}


def _serve(name):
    import jax
    import jax.numpy as jnp
    from generals.agents import (BossAgent, CastleEconomistAgent, DeathtouchClockAgent, DrawGrinderAgent,
                                 FogScoutAgent, HumanExeAgent, RaiderAgent)
    from generals.core.observation import Observation

    factories = {"human_exe": HumanExeAgent, "boss": BossAgent, "castle_economist": CastleEconomistAgent,
                 "deathtouch_clock": DeathtouchClockAgent, "draw_grinder": DrawGrinderAgent,
                 "fog_scout": FogScoutAgent, "raider": RaiderAgent}
    stdin = _sys.stdin
    hs = stdin.readline()
    if not hs:
        return
    pid, H, W = map(int, hs.split())
    agent = factories[name]()
    stateful = name == "human_exe"
    if stateful:
        from generals.agents.human_exe_agent import init_human_exe_memory
        memory = init_human_exe_memory(H, W)
        board_mask = jnp.ones((H, W), dtype=jnp.bool_)
    key = jax.random.PRNGKey(pid)
    while True:
        head = stdin.readline()
        if not head:
            return
        turn, ml, ma, ol, oa = map(int, head.split())
        grids = []
        for _ in range(3):
            grids.append(jnp.asarray([[int(v) for v in stdin.readline().split()] for _ in range(H)], dtype=jnp.int32))
        types, owners, armies = grids
        visible = (types != 0) & (types != 5)
        obs = Observation(
            armies=armies, generals=types == 4, castles=types == 3, mountains=types == 2,
            neutral_cells=(owners == 0) & visible & (types != 2), owned_cells=owners == 1,
            opponent_cells=owners == 2, fog_cells=types == 0, structures_in_fog=types == 5,
            owned_land_count=jnp.int32(ml), owned_army_count=jnp.int32(ma),
            opponent_land_count=jnp.int32(ol), opponent_army_count=jnp.int32(oa), timestep=jnp.int32(turn))
        key, k = jax.random.split(key)
        if stateful:
            action, memory = agent.act_with_memory(obs, k, board_mask, memory)
        else:
            action = agent.act(obs, k)
        _sys.stdout.write(" ".join(str(int(x)) for x in action) + "\n")
        _sys.stdout.flush()


if __name__ == "__main__":
    _serve(_sys.argv[1] if len(_sys.argv) > 1 else "human_exe")
else:
    _s = _u.spec_from_file_location("_ext_stdio_humanexe", _os.path.join(_HERE, "ext_stdio.py"))
    _m = _u.module_from_spec(_s)
    _s.loader.exec_module(_m)
    _inner = []

    def act(obs):
        if not _inner:
            _inner.append(_m.make_act([_PY, "-u", _os.path.abspath(__file__), PARAMS["agent"]], cwd=_ROOT, env=_ENV))
        return _inner[0](obs)

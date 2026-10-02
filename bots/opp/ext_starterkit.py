"""Generic in-process adapter for bots written against the official Python starter kit:
an agent.py with class Agent(player_id, H, W) whose act(obs) takes the starter-kit
Observation dataclass (H, W, turn, my_land, my_army, opp_land, opp_army, type_grid,
owner_grid, army_grid; same codes as our dict) and returns (pass/kind, row, col, dir, split).

    act = ext_starterkit.make_act("/abs/path/agent.py", env={"VAR": None})

agent.py is loaded by file path under a unique module name (each wrapper import gets a
fresh copy, so module-level state of the original is reset every game, as in a fresh
process). A new Agent is created at the first call of a game or when the turn goes
backwards; act.module is the loaded original. quiet=True replaces its print(). env: variables to set (str) or remove (None) before loading, for originals
that read os.environ at import/construction time. LOCAL TESTING ONLY.
"""
import importlib.util
import itertools
import os
import sys
from types import SimpleNamespace

_COUNT = itertools.count()


class _Null:
    def write(self, *a):
        return 0

    def flush(self):
        pass


class _QuietSys:
    """Stands in for `sys` inside the original module: stderr goes nowhere."""
    stderr = _Null()

    def __getattr__(self, k):
        return getattr(sys, k)


def make_act(agent_path, env=None, extra_sys_path=None, quiet=True):
    for k, v in (env or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if extra_sys_path and extra_sys_path not in sys.path:
        sys.path.insert(0, extra_sys_path)
    spec = importlib.util.spec_from_file_location(f"_ext_sk_{os.getpid()}_{next(_COUNT)}", agent_path)
    mod = importlib.util.module_from_spec(spec)
    if quiet:  # silence the original's telemetry prints
        mod.__dict__["print"] = lambda *a, **k: None
    spec.loader.exec_module(mod)
    if quiet and getattr(mod, "sys", None) is sys:
        mod.sys = _QuietSys()
    state = {"agent": None, "turn": -1}

    def act(obs):
        t = obs["turn"]
        if state["agent"] is None or t <= state["turn"]:
            state["agent"] = mod.Agent(obs["player_id"], obs["height"], obs["width"])
        state["turn"] = t
        o = SimpleNamespace(H=obs["height"], W=obs["width"], turn=t, my_land=obs["my_land"],
                            my_army=obs["my_army"], opp_land=obs["opp_land"], opp_army=obs["opp_army"],
                            type_grid=obs["type"], owner_grid=obs["owner"], army_grid=obs["army"])
        return [int(v) for v in state["agent"].act(o)]

    act.module = mod
    return act

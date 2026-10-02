"""External sparring partner: Amin-Debabeche self-play PPO CNN (pure NumPy).

Source: https://github.com/Amin-Debabeche/generals-bots @ 34506fe2d684f163ada3a0cc3401d192436bfc37
        competition/agents/my_bot (HEAD weights == bot_archive/my_bot_main8_iter160.zip)
        plus bot_archive/my_bot_main7_iter79.zip, my_bot_main7_iter308.zip (same code, other weights).
Licence: MIT (engine fork). LOCAL TESTING ONLY, never part of the submission.

Re-implements the original agent.py Agent.act (greedy masked argmax + memory
features) by calling the original numpy_infer.py, loaded by file path so the
module name cannot clash. PARAMS["ckpt"] picks the checkpoint.
"""
import importlib.util
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # multi-threaded BLAS is ~10x slower on a busy box
import numpy as np

try:  # numpy may already be imported by the host process
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

PARAMS = {"ckpt": "main8_iter160"}

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vendor", "ext",
                     "Amin-Debabeche_generals-bots", "competition", "agents")
_DIRS = {
    "head": os.path.join(_ROOT, "my_bot"),
    "main8_iter160": os.path.join(_ROOT, "bot_archive", "main8_iter160"),
    "main7_iter79": os.path.join(_ROOT, "bot_archive", "main7_iter79"),
    "main7_iter308": os.path.join(_ROOT, "bot_archive", "main7_iter308"),
}
PASS = [1, 0, 0, 0, 0]

# process-wide cache (the arena re-imports this file every game)
_CACHE = sys.__dict__.setdefault("_porter_amin_cache", {})


def _load(ckpt):
    if ckpt in _CACHE:
        return _CACHE[ckpt]
    d = _DIRS[ckpt]
    if not os.path.exists(os.path.join(d, "weights.npz")):  # unpack archived checkpoint on demand
        import zipfile
        os.makedirs(d, exist_ok=True)
        with zipfile.ZipFile(os.path.join(_ROOT, "bot_archive", f"my_bot_{ckpt}.zip")) as z:
            z.extractall(d)
    spec = importlib.util.spec_from_file_location(f"_amin_ninf_{ckpt}", os.path.join(d, "numpy_infer.py"))
    ninf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ninf)
    weights, meta = ninf.load_weights(os.path.join(d, "weights.npz"), os.path.join(d, "weights_meta.json"))
    _CACHE[ckpt] = (ninf, weights, meta)
    return _CACHE[ckpt]


_state = {"mem": None, "turn": -1}


def act(obs):
    ninf, weights, meta = _load(PARAMS["ckpt"])
    G = meta["grid_size"]
    H, W, t = obs["height"], obs["width"], obs["turn"]
    if _state["mem"] is None or t <= _state["turn"]:
        _state["mem"] = ninf.init_memory(G)
    _state["turn"] = t
    try:
        raw = ninf.build_obs_tensor(obs["type"], obs["owner"], obs["army"], obs["my_land"],
                                    obs["my_army"], obs["opp_land"], obs["opp_army"], t, G)
        mask = ninf.compute_legal_action_mask(raw[5], raw[0], raw[3], raw[2], raw[1], True)
        normalized = ninf.normalize_obs_tensor(raw.copy(), meta)
        net_in = np.concatenate([normalized, ninf.memory_to_channels(_state["mem"])], axis=0)
        logits = ninf.forward_policy_logits(weights, net_in)
        idx = ninf.greedy_masked_argmax(logits, mask)
        action = ninf.decode_action_index(idx, G, G)
        _state["mem"] = ninf.update_memory(_state["mem"], raw, action)
        p, r, c, d, s = (int(v) for v in action)
        if p != 1 and (r >= H or c >= W):
            return list(PASS)
        return [p, r, c, d, s]
    except Exception as e:  # mirror the original: pass on internal failure
        print(f"[ext_amin] act() failed, passing: {e!r}", file=sys.stderr)
        return list(PASS)

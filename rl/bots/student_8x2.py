"""RL track: small BC student playing through act(obs) (torch, local testing only). NET_FILE set below."""
NET_FILE = "ResBot_8x2.pt"

import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "learn"))
import bc_features as F  # noqa: E402
from bc_train import Net  # noqa: E402

torch.set_num_threads(1)
_NET = None
_TRK = None
_RNG = random.Random(7)
TEMP = float(os.environ.get("STUDENT_TEMP", "0"))


def _net():
    global _NET
    if _NET is None:
        path = os.path.join(ROOT, "rl", "students", NET_FILE)
        cache = F.__dict__.setdefault("_NET_CACHE", {})   # process-wide (bc_features stays in sys.modules)
        if path in cache:
            _NET = cache[path]
            return _NET
        ck = torch.load(path, map_location="cpu")
        net = Net(ch=ck["ch"], blocks=ck["blocks"])
        net.load_state_dict(ck["state"])
        net.eval()
        _NET = net
        cache[path] = net
    return _NET


def act(obs):
    global _TRK
    H, W = int(obs["height"]), int(obs["width"])
    t = int(obs["turn"])
    if _TRK is None or t == 0:
        _TRK = F.Tracker(H, W)
    planes, scal, (O, A, T) = _TRK.update(obs)
    mask = F.legal_mask(O, A, T, H, W, planes)
    with torch.no_grad():
        lg, _ = _net()(torch.from_numpy(planes[None]), torch.from_numpy(scal[None]))
    lg = lg[0].numpy()
    lg[~mask] = -1e9
    if TEMP <= 0:
        idx = int(lg.argmax())
    else:
        z = (lg - lg.max()) / TEMP
        p = np.exp(z)
        p /= p.sum()
        idx = int(np.searchsorted(np.cumsum(p), _RNG.random()))
        idx = min(idx, len(p) - 1)
    a = F.index_action(idx)
    return [int(v) for v in a]

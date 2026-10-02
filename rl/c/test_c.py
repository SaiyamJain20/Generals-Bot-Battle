"""Track C checks.
1) participant_c + F2 params (RL off) plays move-identical to F2 on several games.
2) with exploration on, traces are recorded and games finish without errors.
3) the GRPO/PPO surrogate gradient matches finite differences.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rl", "c"))
import worker as Wk  # noqa: E402

F2 = os.path.join(ROOT, "bots", "versions", "F2.py")
PC = os.path.join(ROOT, "rl", "c", "participant_c.py")


def f2_params():
    import importlib.util
    s = importlib.util.spec_from_file_location("f2p", F2)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return dict(m.PARAMS)


def test_identical(games=4):
    p = f2_params()
    for k in range(games):
        for opp in ("bots/opp/zoo_flash.py", "bots/opp/hunter.py"):
            base = dict(bot=F2, opp=os.path.join(ROOT, opp), map_idx=4000 + k, seat=k % 2, params=None,
                        sigma=0, W=8, seed=0, record_actions=True, max_turns=500)
            a = Wk.play_c(dict(base))
            b = Wk.play_c(dict(base, bot=PC, params=p))
            assert a["actions"] == b["actions"], (opp, k, next(i for i, (x, y) in enumerate(zip(a["actions"], b["actions"])) if x != y))
    print("identical: OK")


def test_explore():
    p = f2_params()
    r = Wk.play_c(dict(bot=PC, opp=os.path.join(ROOT, "bots/opp/zoo_flash.py"), map_idx=4100, seat=0, params=p,
                       sigma=0.3, W=8, seed=7, max_turns=1200))
    assert r["error"] is None, r["error"]
    assert len(r["trace"]) > 5 and len(r["trace"][0]["f"]) == 11 and len(r["trace"][0]["a"]) == 8, len(r["trace"])
    print("explore: OK", r["outcome"], r["turns"], len(r["trace"]))


if __name__ == "__main__":
    test_identical()
    test_explore()


def test_grad():
    import numpy as np
    import grpo as G
    rs = np.random.RandomState(0)
    N, Gn, NF = 300, 8, 11
    F = rs.randn(N, NF); F[:, 7] = 1.0
    w = rs.randn(Gn, NF) * 0.1
    w_anchor = w + rs.randn(Gn, NF) * 0.05
    sigma = 0.3
    MU = F @ w.T
    A = MU + rs.randn(N, Gn) * sigma
    ADV = rs.randn(N)
    WT = rs.rand(N)
    arrays = (F, A, MU, ADV, WT)
    w1 = w + rs.randn(Gn, NF) * 0.01   # a point off the behaviour policy so ratios != 1
    for clip in (10.0, 0.2):
        L, g, _, _ = G.surrogate_and_grad(w1, w_anchor, arrays, sigma, clip, 0.05)
        num = np.zeros_like(w1)
        h = 1e-6
        for i in range(Gn):
            for j in range(NF):
                e = np.zeros_like(w1); e[i, j] = h
                num[i, j] = (G.surrogate_and_grad(w1 + e, w_anchor, arrays, sigma, clip, 0.05)[0] -
                             G.surrogate_and_grad(w1 - e, w_anchor, arrays, sigma, clip, 0.05)[0]) / (2 * h)
        err = np.abs(num - g).max() / (np.abs(num).max() + 1e-12)
        assert err < 1e-4, (clip, err)
    print("grad: OK")

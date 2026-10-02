"""Track C checks.
1) participant_c + F2 params (RL off) plays move-identical to F2 on several games.
2) with exploration on, traces are recorded and games finish without errors.
3) the GRPO/PPO surrogate gradient matches finite differences.
"""
import json
import math
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
    for H in (0, 6):
        pol = G.Policy(w, H=H, seed=3)
        if H:
            pol.V = rs.randn(Gn, H) * 0.2
        anc = G.Policy(w + rs.randn(Gn, NF) * 0.05, H=0)
        sigma = 0.3
        MU = pol.mu(F)
        A = MU + rs.randn(N, Gn) * sigma
        ADV = rs.randn(N)
        WT = rs.rand(N)
        arrays = (F, A, MU, ADV, WT)
        th0 = pol.pack() + rs.randn(pol.pack().size) * 0.01
        am = anc.mu(F)
        for clip in (10.0, 0.2):
            pol.unpack(th0.copy())
            L, g, _, _ = G.surrogate_and_grad(pol, am, arrays, sigma, clip, 0.05)
            num = np.zeros_like(th0)
            h = 1e-6
            for i in range(th0.size):
                e = np.zeros_like(th0); e[i] = h
                pol.unpack(th0 + e); lp = G.surrogate_and_grad(pol, am, arrays, sigma, clip, 0.05)[0]
                pol.unpack(th0 - e); lm = G.surrogate_and_grad(pol, am, arrays, sigma, clip, 0.05)[0]
                num[i] = (lp - lm) / (2 * h)
            err = np.abs(num - g).max() / (np.abs(num).max() + 1e-12)
            assert err < 1e-4, (H, clip, err)
    print("grad: OK (linear + MLP)")


def test_mlp_params_match():
    """participant_c's MLP evaluation == learner's Policy.mu on the same features."""
    import numpy as np
    import grpo as G
    import importlib.util
    w = np.random.RandomState(1).randn(8, 11) * 0.1
    pol = G.Policy(w, H=5, seed=2)
    pol.V = np.random.RandomState(4).randn(8, 5) * 0.3
    base = G.load_params(F2)
    p = pol.params(base)
    s = importlib.util.spec_from_file_location("pcm", PC)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.PARAMS.update(p)
    f = list(np.random.RandomState(5).uniform(-1, 1, 11)); f[7] = 1.0

    class Fake:
        MOD_GROUPS = m.Bot.MOD_GROUPS if hasattr(m, "Bot") else None
    bot_cls = [v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "compute_mods")][0]
    b = bot_cls.__new__(bot_cls)
    b.mod_features = lambda: tuple(f)
    b.turn = 100
    b.compute_mods()
    want = pol.mu(np.array([f]))[0]
    got = [math.log(b.mod[g]) for g in G.GROUPS]
    for x, y in zip(want, got):
        assert abs(max(-1.5, min(1.5, x)) - y) < 1e-4, (want, got)
    print("mlp params match: OK")


if __name__ == "__main__":
    test_identical()
    test_explore()
    test_grad()
    test_mlp_params_match()

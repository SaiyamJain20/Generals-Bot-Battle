"""Unit checks for RO-PPO.  Run: PYTHONPATH=vendor/generals-bots:. .venv312/bin/python rl/hier/test_ro.py
(a) RO_W=None => identical moves to tune_base12g;  (b) reference weights => argmax==heuristic;
(c) GAE / semi-MDP against a hand-computed example;  (d) shaping telescopes exactly (discounted identity).
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE, os.path.join(ROOT, "arena")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from sim import engine as E  # noqa: E402
import run as AR  # noqa: E402
import ro_core  # noqa: E402
import ro_worker  # noqa: E402

BASE = os.path.join(ROOT, "bots/versions/tune_base12g.py")
OPP = os.path.join(ROOT, "bots/opp/zoo_mixed.py")


def fix_budget(m):
    m.PARAMS["soft_budget_ms"] = 10 ** 7
    m.PARAMS["first_budget_ms"] = 10 ** 7
    m.PARAMS["open_plan_s"] = 10 ** 6
    return m


def play_log(mod0, map_idx, turns=250):
    m = AR.maps()[map_idx]
    s = E.from_grid(m["grid"])
    opp = AR.load_bot(OPP)
    if hasattr(opp, "PARAMS") and "soft_budget_ms" in opp.PARAMS:
        fix_budget(opp)
    mods = [mod0, opp]
    log = []
    while not s.done and s.time < turns:
        acts = []
        for p in (0, 1):
            a = mods[p].act(E.observe(s, p))
            if p == 0:
                log.append(tuple(a))
            acts.append(list(a))
        E.step(s, acts)
    return log


def test_a_identical_moves():
    for mp_ in (3, 11, 25):
        base = play_log(fix_budget(AR.load_bot(BASE)), mp_)
        ro = fix_budget(AR.load_bot(ro_worker.BOT_RO))
        assert ro.RO_W is None
        assert play_log(ro, mp_) == base, "RO_W=None differs from heuristic on map %d" % mp_
    print("(a) ok: RO_W=None identical moves on 3 games (250 turns each)")


def test_b_reference_weights():
    import ro_ppo
    for hidden in (0, 16):
        w = ro_ppo.default_weights(hidden)
        n = agree = 0
        for mp_ in (3, 11, 25):
            base = play_log(fix_budget(AR.load_bot(BASE)), mp_)
            rec = []
            ro = fix_budget(AR.load_bot(ro_worker.BOT_RO))
            ro.RO_W, ro.RO_RECORD = w, rec
            log = play_log(ro, mp_)
            n += len(rec)
            agree += sum(1 for r in rec if r["a"] == 0)
            assert all(abs(r["logp"] - r["logp_ref"]) < 1e-9 for r in rec)
            assert log == base
        assert n > 100 and agree / n >= 0.99, (agree, n)
        print("(b) ok hidden=%d: argmax==heuristic on %d/%d decisions; moves identical" % (hidden, agree, n))


def test_b2_noncycle_commit():
    """A forced non-top choice must still play a legal game with consistent cycle state."""
    import ro_ppo
    w = ro_ppo.default_weights(0)
    w["b"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    w["alpha"] = 0.0  # ignore scores
    w["b"][1] = 5.0   # strongly prefer 'garrison'... then capture etc by label bias
    rec = []
    ro = fix_budget(AR.load_bot(ro_worker.BOT_RO))
    ro.RO_W, ro.RO_RECORD, ro.RO_SAMPLE = w, rec, True
    play_log(ro, 7, turns=200)
    assert any(r["a"] != 0 for r in rec) and len(rec) > 20
    print("(b2) ok: non-top choices executed (%d decisions, %d non-top)" % (len(rec), sum(r["a"] != 0 for r in rec)))


def test_c_gae():
    # 3 decisions at turns 0, 10, 12; V = [0.2, 0.5, -0.1]; no shaping; terminal reward +1 at last
    gamma, lam = 0.9, 0.5
    tau = np.array([10, 2, 1.0])
    V = np.array([0.2, 0.5, -0.1])
    R = np.array([0.0, 0.0, 1.0])
    adv, ret = ro_core.gae(R, V, tau, gamma, lam)
    d2 = 1.0 + 0.0 - (-0.1)                      # = 1.1
    d1 = 0.0 + gamma ** 2 * (-0.1) - 0.5         # = -0.581
    d0 = 0.0 + gamma ** 10 * 0.5 - 0.2           # = 0.5*0.3486784401-0.2
    a2 = d2
    a1 = d1 + gamma ** 2 * lam * a2
    a0 = d0 + gamma ** 10 * lam * a1
    assert np.allclose(adv, [a0, a1, a2]), (adv, [a0, a1, a2])
    assert np.allclose(ret, adv + V)
    # credit assignment: a +1 at the end gives a positive advantage to earlier steps with zero-value critic
    adv0, _ = ro_core.gae(np.array([0, 0, 0, 1.0]), np.zeros(4), np.ones(4), 0.99, 0.95)
    assert adv0[0] > 0 and adv0[0] < adv0[3] and np.all(np.diff(adv0) > 0)
    # hand-checked numbers: gamma=lam=1, V=0 -> every advantage equals the terminal reward
    a, _ = ro_core.gae(np.array([0, 0, 1.0]), np.zeros(3), np.ones(3), 1.0, 1.0)
    assert np.allclose(a, 1.0)
    # terminal gap: decisions at turns 0 and 30, game ends at t_end=40 (last decision 10 turns before end)
    g, lam_, beta = 0.99, 0.9, 0.2
    cn = [(10, 10, 5, 5), (30, 10, 5, 5)]                 # Phi_1 = beta*(0.5*tanh(ln3)+0) ; Phi_0 = 0
    ph1 = beta * 0.5 * math.tanh(math.log(3.0))
    V = np.array([0.1, 0.3])
    adv_t, _, Rv, tauv = ro_core.episode_advantages([0, 30], cn, V, 1.0, g, lam_, beta, t_end=40)
    assert np.allclose(tauv, [30, 10])
    R1 = -ph1 + g ** 9 * 1.0                               # outcome discounted by gamma^(t_end-1-30)
    R0 = g ** 30 * ph1 - 0.0
    assert np.allclose(Rv, [R0, R1]), (Rv, [R0, R1])
    d1 = R1 + 0.0 - 0.3
    d0 = R0 + g ** 30 * 0.3 - 0.1
    assert np.allclose(adv_t, [d0 + g ** 30 * lam_ * d1, d1])
    # with t_end=None the old behaviour (gap 0, tau 1) is recovered
    _, _, Rn, taun = ro_core.episode_advantages([0, 30], cn, V, 1.0, g, lam_, beta)
    assert np.allclose(Rn[1], -ph1 + 1.0) and taun[1] == 1
    print("(c) ok: semi-MDP GAE matches hand computation (a0=%.6f a1=%.6f a2=%.6f)" % (a0, a1, a2))


def test_d_shaping_telescopes():
    rng = random.Random(1)
    for trial in range(20):
        n = rng.randint(1, 60)
        turns = sorted(rng.sample(range(0, 1200), n))
        cnts = [(rng.randint(1, 500), rng.randint(1, 500), rng.randint(1, 200), rng.randint(1, 200)) for _ in range(n)]
        gamma, beta = 0.998, rng.choice([0.2, 0.1, 1.0])
        F, tau = ro_core.shaping(turns, cnts, gamma, beta)
        # discount each F_k back to decision 0:  gamma^(turn_k - turn_0)
        lhs = sum(gamma ** (turns[k] - turns[0]) * F[k] for k in range(n))
        rhs = 0.0 - ro_core.potential(cnts[0], beta)      # Phi(terminal) - Phi(s0)
        assert abs(lhs - rhs) < 1e-12, (lhs, rhs)
    # undiscounted identity (gamma=1): plain sum of shaped rewards = -Phi(s0)
    F, _ = ro_core.shaping([0, 5, 9], [(10, 10, 5, 5), (20, 10, 6, 5), (5, 30, 3, 9)], 1.0, 0.2)
    assert abs(F.sum() + ro_core.potential((10, 10, 5, 5), 0.2)) < 1e-12
    # real episode
    rec = []
    ro = fix_budget(AR.load_bot(ro_worker.BOT_RO))
    import ro_ppo
    ro.RO_W, ro.RO_RECORD = ro_ppo.default_weights(0), rec
    play_log(ro, 5, turns=120)
    turns, cnts = [r["turn"] for r in rec], [r["cnt"] for r in rec]
    F, _ = ro_core.shaping(turns, cnts, 0.998, 0.2)
    lhs = sum(0.998 ** (turns[k] - turns[0]) * F[k] for k in range(len(rec)))
    assert abs(lhs + ro_core.potential(cnts[0], 0.2)) < 1e-12
    print("(d) ok: discounted shaping telescopes to -Phi(s0) (20 random + 1 real episode, %d decisions)" % len(rec))


def test_e_unseen_label_and_overhead():
    import time
    import ro_ppo
    ro = fix_budget(AR.load_bot(ro_worker.BOT_RO))
    assert ro._ro_lab("cyc_mystery") == len(ro.RO_LABELS) - 1 and ro._ro_lab("cyc_castle_g") == ro._ro_lab("cyc_castle")
    ro.RO_W = ro_ppo.default_weights(16)
    ro.RO_W["u"][0] = [0.1] * 24
    ro.RO_W["Uh"][0] = [0.1] * 16
    # measure overhead of _ro_pick in a live game
    tot = [0.0, 0]
    orig = ro._ro_pick

    def timed(*a):
        t0 = time.perf_counter()
        r = orig(*a)
        tot[0] += time.perf_counter() - t0
        tot[1] += 1
        return r
    ro._ro_pick = timed
    play_log(ro, 9, turns=300)
    ms = 1000 * tot[0] / max(1, tot[1])
    print("(e) ok: unseen labels map to 'other'; _ro_pick overhead %.3f ms/decision over %d decisions" % (ms, tot[1]))
    assert ms < 1.0


if __name__ == "__main__":
    test_c_gae()
    test_d_shaping_telescopes()
    test_e_unseen_label_and_overhead()
    test_a_identical_moves()
    test_b_reference_weights()
    test_b2_noncycle_commit()
    print("ALL OK")

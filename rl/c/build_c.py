"""Track C: build rl/c/participant_c.py from the master heuristic (bots/participant.py).

Changes (all inert with zero new weights, so participant_c + F2 params == F2):
  * the context modulator gets an 8th group "build" (castle-build option weight) and 3 extra
    features (index 8..10): distance to the enemy general belief, enemy presence near home,
    garrison pressure. Weights m_<group>_<j> for the new slots default to 0.
  * a training-only exploration/trace hook: module global RL (None in deployment). When set to
    {"rng": random.Random, "sigma": float, "W": int, "trace": []}, every W turns each group's
    modulator logit is drawn as a = mu(f) + N(0, sigma^2) and held for the window; the window's
    (turn, features, mu, a) is appended to RL["trace"] for the policy-gradient learner.

    python rl/c/build_c.py [--src /path/to/master/bots/participant.py] [--out rl/c/participant_c.py]
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = "/home/saiyamjain/Desktop/Bot-Battle/bots/participant.py"

GROUPS = ("launch", "cycle", "capture", "garrison", "scout", "home", "kill", "build")
NF = 11  # features: 0 phase, 1 army ratio, 2 land ratio, 3 enemy castles, 4 gather style,
#          5 egen known, 6 threat, 7 bias, 8 enemy distance, 9 enemy near home, 10 garrison pressure


def rep(src, old, new, count=1):
    n = src.count(old)
    if n != count:
        raise SystemExit(f"patch anchor found {n} times (want {count}): {old[:70]!r}")
    return src.replace(old, new)


def build(src):
    # 1) PARAMS: new modulator slots (after the last existing one)
    extra = []
    for g in GROUPS:
        for j in range(NF):
            if g == "build" or j >= 8:
                extra.append(f'    "m_{g}_{j}": 0.0,\n')
    src = rep(src, '    "m_kill_7": 0.0,\n', '    "m_kill_7": 0.0,\n' + "".join(extra))

    # 2) groups
    src = rep(src, 'MOD_GROUPS = ("launch", "cycle", "capture", "garrison", "scout", "home", "kill")',
              'MOD_GROUPS = ("launch", "cycle", "capture", "garrison", "scout", "home", "kill", "build")')

    # 3) features 8..10
    old_ret = "        return (t / 600.0 - 1.0, f1, f2, f3, f4, f5, f6, 1.0)\n"
    new_ret = (
        "        # Track C features: distance to the enemy general belief, enemy presence near home,\n"
        "        # garrison pressure (need from the previous turn vs army on the general)\n"
        "        dg = self.dist_g\n"
        "        if self.egen >= 0:\n"
        "            de = min(dg[self.egen], 60)\n"
        "        else:\n"
        "            cs = self.cands or self.cands0\n"
        "            de = sum(min(dg[c], 60) for c in cs) / len(cs) if cs else 22.0\n"
        "        f8 = max(-1.0, min(1.0, (de - 22.0) / 10.0))\n"
        "        O = self.O\n"
        "        near = sum(1 for i in range(self.n) if O[i] == 2 and dg[i] <= 8)\n"
        "        f9 = min(near, 12) / 6.0 - 1.0\n"
        "        ag = self.A[self.general] if self.general >= 0 else 1\n"
        "        f10 = math.tanh((getattr(self, 'need_g', 0) - ag) / max(10.0, ag))\n"
        "        return (t / 600.0 - 1.0, f1, f2, f3, f4, f5, f6, 1.0, f8, f9, f10)\n")
    src = rep(src, old_ret, new_ret)

    # 4) compute_mods: variable feature count, exploration/trace hook
    old_cm = """            f = (0.0,) * 7 + (1.0,)
        for gname in self.MOD_GROUPS:
            z = 0.0
            for j in range(8):
                z += P["m_%s_%d" % (gname, j)] * f[j]
            mod[gname] = math.exp(max(-1.5, min(1.5, z)))
"""
    new_cm = """            f = (0.0,) * 7 + (1.0, 0.0, 0.0, 0.0)
        mu = {}
        for gname in self.MOD_GROUPS:
            z = 0.0
            for j in range(len(f)):
                z += P.get("m_%s_%d" % (gname, j), 0.0) * f[j]
            mu[gname] = z
        nh = int(P.get("mh_n", 0))
        if nh > 0:
            # nonlinear part: z_g += sum_k v[g][k] * tanh(u[k] . f + c[k])
            mlp = getattr(self, "_mlp", None)
            if mlp is None:
                U = [[P.get("mu_%d_%d" % (k, j), 0.0) for j in range(len(f))] for k in range(nh)]
                C = [P.get("mc_%d" % k, 0.0) for k in range(nh)]
                V = {g: [P.get("mv_%s_%d" % (g, k), 0.0) for k in range(nh)] for g in self.MOD_GROUPS}
                mlp = self._mlp = (U, C, V)
            U, C, V = mlp
            h = [math.tanh(C[k] + sum(U[k][j] * f[j] for j in range(len(f)))) for k in range(nh)]
            for gname in self.MOD_GROUPS:
                vg = V[gname]
                mu[gname] += sum(vg[k] * h[k] for k in range(nh))
        rl = RL
        if rl is not None:
            # training only: Gaussian exploration in logit space, held for a window of W turns
            if rl.get("win") is None or self.turn >= rl["win_end"]:
                a = {g: mu[g] + rl["rng"].gauss(0.0, rl["sigma"]) for g in self.MOD_GROUPS}
                rl["win"] = {"t": self.turn, "f": list(f), "mu": mu, "a": a}
                rl["win_end"] = self.turn + rl["W"]
                rl["trace"].append(rl["win"])
            a = rl["win"]["a"]
            for gname in self.MOD_GROUPS:
                mod[gname] = math.exp(max(-1.5, min(1.5, a[gname])))
            return
        for gname in self.MOD_GROUPS:
            mod[gname] = math.exp(max(-1.5, min(1.5, mu[gname])))
"""
    src = rep(src, old_cm, new_cm)

    # 5) build option uses the build multiplier
    src = rep(src, '        return (P["w_build"], [2, best // self.W, best % self.W, 0, 0])',
              '        return (P["w_build"] * self.mod.get("build", 1.0), [2, best // self.W, best % self.W, 0, 0])')

    # 6) module global hook
    src = rep(src, "DEBUG = False  # arena sets True to surface exceptions\n",
              "DEBUG = False  # arena sets True to surface exceptions\n"
              "RL = None  # training-only exploration/trace hook (Track C); None in the submission\n")
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=MASTER)
    ap.add_argument("--out", default=os.path.join(HERE, "participant_c.py"))
    args = ap.parse_args()
    out = build(open(args.src).read())
    with open(args.out, "w") as f:
        f.write(out)
    print(f"wrote {args.out} ({len(out)} bytes) from {args.src}")


if __name__ == "__main__":
    main()

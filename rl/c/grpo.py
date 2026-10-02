"""Track C learner: GRPO / PPO over the context modulator of the heuristic (participant_c.py).

Policy (per decision window of W turns):  a_g ~ N(mu_g(f), sigma^2),  mu_g(f) = sum_j w[g][j] * f_j,
for 8 option groups g and 11 state features f (phase, army/land ratio, enemy castles, enemy style,
general known, threat, bias, enemy distance, enemy near home, garrison pressure). The heuristic turns
exp(clip(a_g)) into option-weight multipliers; deployment uses the mean (a = mu, no noise).

Objective (RLVR-style: the reward is the exact engine's verdict):
  R = outcome (+1 win, -1 loss, draw = --draw) + c_margin * tanh(0.5 * log(my_army / opp_army) at the end)
  GRPO advantage: K rollouts of the same (map, seat, opponent) group with different noise;
  A_k = R_k - mean_group(R), then normalised over the batch (no learned critic).
  Every window of game k carries A_k, weighted 1/n_windows(k) so each game counts once.
  PPO-clip surrogate on the per-window likelihood ratio, several epochs per batch, plus a KL anchor
  to the starting policy (es4 / F2):  beta * (mu - mu_anchor)^2 / (2 sigma^2).
Opponents: PFSP over a fixed pool + frozen self snapshots. Gradients are analytic (Gaussian policy).

    python rl/c/grpo.py --out rl/c/runs/c1 --hours 7 --workers 36 --groups 24 --K 4
"""
import argparse
import faulthandler
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import worker as Wk  # noqa: E402

GROUPS = ("launch", "cycle", "capture", "garrison", "scout", "home", "kill", "build")
NF = 11
PC = os.path.join(HERE, "participant_c.py")

DEFAULT_POOL = [
    "bots/versions/F2.py", "rl/c/opp/F0.py", "bots/versions/t1c.py", "bots/versions/c_a2es3.py",
    "bots/opp/bc_resbot96.py", "bots/opp/bc_resbot128.py", "bots/opp/bc_nanomena96.py", "bots/opp/bc_kubic96.py",
    "bots/opp/ext_ksolmann.py",
    "bots/opp/zoo_flash.py", "bots/opp/zoo_mixed.py", "bots/opp/zoo_sniper.py", "bots/opp/rusher.py",
    "bots/opp/hunter.py",
]
EVAL_SET = ["bots/versions/F2.py", "bots/opp/bc_resbot96.py", "bots/opp/bc_nanomena96.py",
            "bots/opp/ext_ksolmann.py", "bots/versions/t1c.py", "bots/opp/hunter.py"]


def load_params(path):
    import importlib.util
    s = importlib.util.spec_from_file_location("cp_%d" % random.randrange(10 ** 9), path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return dict(m.PARAMS)


def w_from_params(p):
    w = np.zeros((len(GROUPS), NF))
    for gi, g in enumerate(GROUPS):
        for j in range(NF):
            w[gi, j] = float(p.get("m_%s_%d" % (g, j), 0.0))
    return w


def params_from_w(base, w):
    p = dict(base)
    for gi, g in enumerate(GROUPS):
        for j in range(NF):
            p["m_%s_%d" % (g, j)] = round(float(w[gi, j]), 5)
    return p


class Policy:
    """mu(f) = W f + V tanh(U f + c); H = 0 means linear only."""

    def __init__(self, w, H=0, seed=0, u_scale=0.3):
        G, F = w.shape
        rs = np.random.RandomState(seed)
        self.H = H
        self.W = w.copy()
        self.U = rs.randn(H, F) * u_scale if H else np.zeros((0, F))
        self.c = np.zeros(H)
        self.V = np.zeros((G, H))

    def pack(self):
        return np.concatenate([self.W.ravel(), self.U.ravel(), self.c, self.V.ravel()])

    def unpack(self, th):
        G, F = self.W.shape
        H = self.H
        i = 0
        self.W = th[i:i + G * F].reshape(G, F); i += G * F
        self.U = th[i:i + H * F].reshape(H, F); i += H * F
        self.c = th[i:i + H]; i += H
        self.V = th[i:i + G * H].reshape(G, H)

    def mu(self, F):
        out = F @ self.W.T
        if self.H:
            h = np.tanh(F @ self.U.T + self.c)
            out = out + h @ self.V.T
        return out

    def grad(self, F, g_mu):
        """Chain rule: g_mu (N, G) = dL/dmu -> dL/dtheta (packed)."""
        gW = g_mu.T @ F
        if not self.H:
            return np.concatenate([gW.ravel()])
        h = np.tanh(F @ self.U.T + self.c)
        gV = g_mu.T @ h
        gpre = (g_mu @ self.V) * (1 - h * h)
        gU = gpre.T @ F
        gc = gpre.sum(0)
        return np.concatenate([gW.ravel(), gU.ravel(), gc, gV.ravel()])

    def params(self, base):
        p = params_from_w(base, self.W)
        if self.H:
            p["mh_n"] = self.H
            for k in range(self.H):
                p["mc_%d" % k] = round(float(self.c[k]), 5)
                for j in range(self.U.shape[1]):
                    p["mu_%d_%d" % (k, j)] = round(float(self.U[k, j]), 5)
                for gi, g in enumerate(GROUPS):
                    p["mv_%s_%d" % (g, k)] = round(float(self.V[gi, k]), 5)
        return p


def write_snapshot(path, params):
    """A frozen opponent file: participant_c with these params baked in (RL hook off)."""
    src = open(PC).read()
    k = src.index("\nDIRS = ")
    code = src[:k] + "\nPARAMS.update(" + repr(params) + ")\n" + src[k:]
    with open(path, "w") as f:
        f.write(code)
    return path


def _run(task):
    try:
        return Wk.play_c(task)
    except Exception as e:  # never kill the pool
        return {"error": f"{type(e).__name__}: {e}", "trace": [], "outcome": None, "opp": task.get("opp"),
                "map": task.get("map_idx"), "seat": task.get("seat")}


class Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, w, g):  # ascent
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return w + self.lr * mh / (np.sqrt(vh) + self.eps)


def batch_arrays(results, groups_of, args):
    """Turn rollout results into (F, A, MU_old, ADV, WEIGHT) arrays."""
    # rewards and group-relative advantages
    for r in results:
        o = r["outcome"]
        if o == 0.0:
            o = -args.draw
        r["R"] = o + args.margin * math.tanh(0.5 * r.get("army_ratio", 0.0))
    adv = {}
    for gid, idxs in groups_of.items():
        rs = [results[i]["R"] for i in idxs]
        m = sum(rs) / len(rs)
        for i in idxs:
            adv[i] = results[i]["R"] - m
    vals = np.array(list(adv.values())) if adv else np.zeros(1)
    sd = float(vals.std()) + 1e-6
    F, A, MU, ADV, WT = [], [], [], [], []
    for i, r in enumerate(results):
        if i not in adv or not r["trace"]:
            continue
        a_i = max(-3.0, min(3.0, adv[i] / sd))
        n = len(r["trace"])
        for win in r["trace"]:
            F.append(win["f"])
            A.append([win["a"][g] for g in GROUPS])
            MU.append([win["mu"][g] for g in GROUPS])
            ADV.append(a_i)
            WT.append(1.0 / n)
    if not F:
        return None
    return (np.array(F), np.array(A), np.array(MU), np.array(ADV), np.array(WT))


def surrogate_and_grad(pol, anchor_mu, arrays, sigma, clip, kl):
    """PPO-clip surrogate (to maximise) with the anchor penalty, and its exact gradient (packed)."""
    F, A, MU, ADV, WT = arrays
    s2 = sigma * sigma
    wsum = WT.sum()
    mu_new = pol.mu(F)                                        # (N, G)
    logr = (((A - MU) ** 2) - ((A - mu_new) ** 2)).sum(1) / (2 * s2)
    logr = np.clip(logr, -20, 20)
    r = np.exp(logr)
    rc = np.clip(r, 1 - clip, 1 + clip)
    surr = np.minimum(r * ADV, rc * ADV)
    pen = (((mu_new - anchor_mu) ** 2).sum(1)) / (2 * s2)
    L = float((WT * surr).sum() / wsum - kl * (WT * pen).sum() / wsum)
    clipped = ((ADV > 0) & (r > 1 + clip)) | ((ADV < 0) & (r < 1 - clip))
    coef = np.where(clipped, 0.0, WT * ADV * r)               # d surr / d logr (zero where clipped)
    g_mu = coef[:, None] * (A - mu_new) / s2                  # d logr / d mu_new = (a - mu_new) / s2
    g_mu -= kl * WT[:, None] * (mu_new - anchor_mu) / s2
    grad = pol.grad(F, g_mu) / wsum
    kl_old = float((WT * (((mu_new - MU) ** 2).sum(1) / (2 * s2))).sum() / wsum)
    return L, grad, clipped, kl_old


def ppo_update(pol, anchor, opt, arrays, sigma, args):
    WT = arrays[4]
    anchor_mu = anchor.mu(arrays[0])
    stats = {}
    for ep in range(args.epochs):
        L, grad, clipped, kl_old = surrogate_and_grad(pol, anchor_mu, arrays, sigma, args.clip, args.kl)
        stats = {"surr": round(L, 5), "kl_old": round(kl_old, 5),
                 "clipfrac": round(float((WT * clipped).sum() / WT.sum()), 3),
                 "gnorm": round(float(np.linalg.norm(grad)), 4), "epochs": ep + 1}
        if kl_old > args.target_kl:
            break
        th = opt.step(pol.pack(), grad)
        pol.unpack(np.clip(th, -args.wmax, args.wmax))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="bots/versions/F2.py", help="bot file whose PARAMS seed the policy (es4/F2)")
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=0.35)
    ap.add_argument("--sigma1", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.08)
    ap.add_argument("--kl", type=float, default=0.05, help="anchor penalty weight (to the start policy)")
    ap.add_argument("--wmax", type=float, default=2.0)
    ap.add_argument("--draw", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--pool", nargs="*", default=None)
    ap.add_argument("--league-every", type=int, default=8)
    ap.add_argument("--league-max", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--map-base", type=int, default=2000)
    ap.add_argument("--eval-map-base", type=int, default=17000)
    ap.add_argument("--iter-timeout", type=float, default=3000)
    ap.add_argument("--max-turns", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mlp", type=int, default=0, help="hidden units of the nonlinear term (0 = linear)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    base = load_params(os.path.join(ROOT, args.start))
    for gi, g in enumerate(GROUPS):          # make sure the new slots exist in the base
        for j in range(NF):
            base.setdefault("m_%s_%d" % (g, j), 0.0)
    w0 = w_from_params(base)
    pol = Policy(w0, H=args.mlp, seed=args.seed)
    anchor = Policy(w0, H=0)
    opt = Adam(pol.pack().shape, args.lr)
    pool_paths = [os.path.join(ROOT, p) for p in (args.pool or DEFAULT_POOL)]
    pool_paths = [p for p in pool_paths if os.path.exists(p)]
    score = {p: 0.5 for p in pool_paths}
    league = []
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    pool = Pool(workers, maxtasksperchild=20)
    t0 = time.time()
    t_end = t0 + 3600 * args.hours
    it = 0
    map_ptr = args.map_base
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    json.dump({"args": vars(args), "pool": pool_paths, "w0": w0.tolist()}, open(os.path.join(args.out, "config.json"), "w"))
    while time.time() < t_end:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(args.iter_timeout + 600, repeat=True)
        frac = min(1.0, (time.time() - t0) / (3600 * args.hours))
        sigma = args.sigma0 + (args.sigma1 - args.sigma0) * frac
        # ---- choose opponents (PFSP + 20% uniform) and build groups
        opps_all = pool_paths + league
        wts = [(1.0 - score.get(o, 0.5) + 0.15) ** 2 for o in opps_all]
        tasks, groups_of = [], {}
        policy = pol.params(base)
        for gidx in range(args.groups):
            o = rng.choice(opps_all) if rng.random() < 0.2 else rng.choices(opps_all, weights=wts)[0]
            seat = gidx % 2
            mi = map_ptr + gidx // 2
            for k in range(args.K):
                groups_of.setdefault(gidx, []).append(len(tasks))
                tasks.append(dict(bot=PC, opp=o, map_idx=mi, seat=seat, params=policy, sigma=sigma, W=args.W,
                                  seed=rng.randrange(2 ** 31), max_turns=args.max_turns))
        map_ptr += args.groups // 2 + 1
        ti = time.time()
        try:
            results = pool.map_async(_run, tasks, chunksize=1).get(timeout=args.iter_timeout)
        except Exception as e:
            print(json.dumps({"it": it, "warning": f"pool failure {type(e).__name__}; rebuilding"}), flush=True)
            pool.terminate()
            pool = Pool(workers, maxtasksperchild=20)
            continue
        # drop failed games (and their whole group, to keep the baseline unbiased)
        bad_groups = {gid for gid, idxs in groups_of.items() if any(results[i].get("outcome") is None for i in idxs)}
        errs = [results[i].get("error") for gid in bad_groups for i in groups_of[gid] if results[i].get("error")]
        groups_ok = {gid: idxs for gid, idxs in groups_of.items() if gid not in bad_groups}
        # opponent running scores (EMA) from the learner's perspective
        per = {}
        for gid, idxs in groups_ok.items():
            for i in idxs:
                r = results[i]
                per.setdefault(r["opp"], []).append(0.5 * (r["outcome"] + 1.0))
        for o, v in per.items():
            score[o] = 0.8 * score.get(o, 0.5) + 0.2 * (sum(v) / len(v))
        arrays = batch_arrays(results, groups_ok, args)
        stats = {}
        if arrays is not None:
            stats = ppo_update(pol, anchor, opt, arrays, sigma, args)
        rec = {"it": it, "sec": round(time.time() - ti, 1), "sigma": round(sigma, 3), "games": len(tasks),
               "bad_groups": len(bad_groups), "mean_win": round(sum(sum(v) for v in per.values()) /
                                                                max(1, sum(len(v) for v in per.values())), 3),
               "per_opp": {os.path.basename(o): round(sum(v) / len(v), 2) for o, v in per.items()},
               "windows": 0 if arrays is None else int(len(arrays[0])), **stats,
               "w_drift": round(float(np.abs(pol.W - w0).mean()), 4),
               "v_norm": round(float(np.abs(pol.V).mean()), 4) if pol.H else 0.0}
        if errs:
            rec["err_sample"] = errs[0][:200]
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        json.dump(pol.params(base), open(os.path.join(args.out, "policy_params.json"), "w"), indent=0)
        np.save(os.path.join(args.out, "theta.npy"), pol.pack())
        # ---- league snapshot
        if args.league_every and it % args.league_every == args.league_every - 1:
            path = write_snapshot(os.path.join(args.out, f"snap_{it}.py"), pol.params(base))
            league.append(path)
            score[path] = 0.5
            if len(league) > args.league_max:
                old = league.pop(0)
                score.pop(old, None)
        # ---- deterministic evaluation on held-out maps
        if args.eval_every and it % args.eval_every == args.eval_every - 1:
            etasks = []
            for o in EVAL_SET:
                op = os.path.join(ROOT, o)
                if not os.path.exists(op):
                    continue
                for g in range(args.eval_games):
                    etasks.append(dict(bot=PC, opp=op, map_idx=args.eval_map_base + g // 2, seat=g % 2,
                                       params=pol.params(base), sigma=0, W=args.W, seed=0,
                                       max_turns=args.max_turns))
            try:
                er = pool.map_async(_run, etasks, chunksize=1).get(timeout=args.iter_timeout)
                ev = {}
                for r in er:
                    if r.get("outcome") is not None:
                        ev.setdefault(os.path.basename(r["opp"]), []).append(0.5 * (r["outcome"] + 1.0))
                erec = {"eval_it": it, **{k: round(sum(v) / len(v), 3) for k, v in ev.items()},
                        "pooled": round(sum(sum(v) for v in ev.values()) / max(1, sum(len(v) for v in ev.values())), 3)}
                print(json.dumps(erec), flush=True)
                logf.write(json.dumps(erec) + "\n")
                logf.flush()
                json.dump(pol.params(base), open(os.path.join(args.out, f"policy_it{it}.json"), "w"), indent=0)
            except Exception as e:
                print(json.dumps({"eval_it": it, "warning": f"eval failure {type(e).__name__}"}), flush=True)
                pool.terminate()
                pool = Pool(workers, maxtasksperchild=20)
        it += 1
    pool.close()


if __name__ == "__main__":
    main()

"""RO-PPO: learner + multiprocessing rollout workers for the Residual Option Policy (Track A).

Train:
    PYTHONPATH=vendor/generals-bots:. OMP_NUM_THREADS=1 .venv312/bin/python rl/hier/ro_ppo.py \
        --name run1 --workers 3 --games-per-iter 48 --hours 6
Eval (argmax policy, paired seats):
    ... rl/hier/ro_ppo.py --eval rl/hier/runs/run1/weights_latest.json --eval-opp bots/versions/tune_base12g.py \
        --eval-pairs 40 --eval-maps data/maps_fresh.jsonl --map-offset 1800

Policy: z_i = alpha*s_i + b[lab] + u[lab].phi + v.psi_i + Uh[lab].tanh(W1 phi + b1); critic = MLP(phi) -> win logit.
Weights are exchanged as plain JSON (rl/hier/runs/<name>/weights_<iter>.json) and loaded by bot_ro.RO_W.
"""
import argparse
import collections
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import ro_core  # noqa: E402
import ro_worker  # noqa: E402

NPHI, NPSI, NLAB, KMAX = 24, 8, 9, 8
ALPHA0 = 4.0

DEFAULT_OPPS = ["bots/versions/tune_base12g.py", "bots/versions/t1c.py", "bots/versions/c_a2es3.py",
                "bots/opp/zoo_*.py", "bots/opp/ext_*.py", "bots/opp/bc_*.py"]


# ----------------------------------------------------------------------------- weights
def default_weights(hidden=16, alpha0=ALPHA0, seed=0):
    rng = np.random.RandomState(seed)
    return {"alpha": alpha0, "alpha0": alpha0, "u": [[0.0] * NPHI for _ in range(NLAB)],
            "v": [0.0] * NPSI, "b": [0.0] * NLAB,
            "W1": (rng.randn(hidden, NPHI) * (1.0 / math.sqrt(NPHI))).round(6).tolist() if hidden else [],
            "b1": [0.0] * hidden, "Uh": [[0.0] * hidden for _ in range(NLAB)] if hidden else []}


class ROPolicy(nn.Module):
    def __init__(self, w):
        super().__init__()
        t = lambda x: nn.Parameter(torch.tensor(x, dtype=torch.float32))  # noqa: E731
        self.alpha = t(float(w["alpha"]))
        self.alpha0 = float(w["alpha0"])
        self.u, self.v, self.b = t(w["u"]), t(w["v"]), t(w["b"])
        self.has_h = bool(w.get("W1"))
        if self.has_h:
            self.W1, self.b1, self.Uh = t(w["W1"]), t(w["b1"]), t(w["Uh"])

    def logits(self, phi, psi, s, lab, mask):
        z = self.alpha * s + self.b[lab] + (self.u[lab] * phi[:, None, :]).sum(-1) + (psi * self.v).sum(-1)
        if self.has_h:
            h = torch.tanh(phi @ self.W1.T + self.b1)
            z = z + (self.Uh[lab] * h[:, None, :]).sum(-1)
        return z.masked_fill(~mask, -1e9)

    def logp(self, phi, psi, s, lab, mask):
        return Fn.log_softmax(self.logits(phi, psi, s, lab, mask), dim=-1)

    def logp_ref(self, s, mask):
        return Fn.log_softmax((self.alpha0 * s).masked_fill(~mask, -1e9), dim=-1)

    def to_json(self):
        d = {"alpha": float(self.alpha), "alpha0": self.alpha0, "u": self.u.tolist(),
             "v": self.v.tolist(), "b": self.b.tolist()}
        if self.has_h:
            d.update(W1=self.W1.tolist(), b1=self.b1.tolist(), Uh=self.Uh.tolist())
        return d


class Critic(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(NPHI, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, phi):
        return self.net(phi).squeeze(-1)  # win-probability logit


# ----------------------------------------------------------------------------- batching
def episode_arrays(recs):
    n = len(recs)
    phi = np.zeros((n, NPHI), np.float32)
    psi = np.zeros((n, KMAX, NPSI), np.float32)
    s = np.zeros((n, KMAX), np.float32)
    lab = np.zeros((n, KMAX), np.int64)
    mask = np.zeros((n, KMAX), bool)
    a = np.zeros(n, np.int64)
    for i, r in enumerate(recs):
        k = len(r["s"])
        phi[i] = r["phi"]
        psi[i, :k] = r["psi"]
        s[i, :k] = r["s"]
        lab[i, :k] = r["lab"]
        mask[i, :k] = True
        a[i] = r["a"]
    return dict(phi=phi, psi=psi, s=s, lab=lab, mask=mask, a=a,
                logp=np.array([r["logp"] for r in recs], np.float32),
                turns=[r["turn"] for r in recs], cnts=[r["cnt"] for r in recs])


def build_batch(episodes, critic, gamma, lam, beta, r_draw):
    """episodes: list of (recs, outcome in {-1,0,1}). Returns dict of torch tensors and stats."""
    parts = {k: [] for k in ("phi", "psi", "s", "lab", "mask", "a", "logp", "adv", "ret", "tgt")}
    for recs, out, t_end in episodes:
        if not recs:
            continue
        e = episode_arrays(recs)
        with torch.no_grad():
            v = torch.tanh(critic(torch.from_numpy(e["phi"])) / 2.0).numpy().astype(np.float64)
        R_term = {1: 1.0, -1: -1.0, 0: r_draw}[out]
        adv, ret, _, _ = ro_core.episode_advantages(e["turns"], e["cnts"], v, R_term, gamma, lam, beta, t_end)
        for k in ("phi", "psi", "s", "lab", "mask", "a", "logp"):
            parts[k].append(e[k])
        parts["adv"].append(adv.astype(np.float32))
        parts["ret"].append(ret.astype(np.float32))
        parts["tgt"].append(np.full(len(recs), (R_term + 1.0) / 2.0, np.float32))
    if not parts["phi"]:
        return None
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}


# ----------------------------------------------------------------------------- league
def resolve_opps(patterns):
    out, seen = [], set()
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))) if any(c in pat for c in "*?[") \
                else [os.path.join(ROOT, pat)]:
            if p not in seen and os.path.exists(p):
                seen.add(p)
                out.append({"name": os.path.basename(p)[:-3], "path": p})
    return out


class League:
    def __init__(self, specs, rng, window=100, snaps_keep=6, p_self=0.2, cost_power=1.0, cost_ref=0.012):
        self.fixed = specs
        self.cost = {}          # opponent -> EMA wall seconds per game turn (both bots)
        self.cost_power, self.cost_ref = cost_power, cost_ref
        self.rng = rng
        self.win = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self.snaps = collections.deque(maxlen=snaps_keep)
        self.p_self = p_self

    def add_snap(self, name, w):
        self.snaps.append({"name": name, "snap": w})

    def note_cost(self, name, wall, turns):
        if turns > 20:
            c = wall / turns
            self.cost[name] = c if name not in self.cost else 0.7 * self.cost[name] + 0.3 * c

    def wr(self, name):
        d = self.win[name]
        return sum(d) / len(d) if d else 0.5

    def pool(self):
        return list(self.fixed) + list(self.snaps)

    def sample(self):
        pool = self.pool()
        wts = [((1.0 - self.wr(o["name"])) ** 2 + 0.03) *
               min(1.0, self.cost_ref / max(1e-9, self.cost.get(o["name"], 0.0))) ** self.cost_power
               for o in pool]
        # snapshots share total mass p_self at most (they are mostly near-clones of the learner)
        sn = [i for i, o in enumerate(pool) if "snap" in o]
        if sn:
            tot = sum(wts)
            ssum = sum(wts[i] for i in sn)
            cap = self.p_self * tot / max(1e-9, 1 - self.p_self)
            if ssum > cap:
                for i in sn:
                    wts[i] *= cap / ssum
        return self.rng.choices(pool, weights=wts)[0]


# ----------------------------------------------------------------------------- PPO update
def ppo_update(pol, critic, opt_p, opt_v, batch, args, klc, train_alpha):
    n = batch["phi"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = collections.defaultdict(float)
    cnt = 0
    for _ in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.mb):
            ix = perm[i:i + args.mb]
            lp_all = pol.logp(batch["phi"][ix], batch["psi"][ix], batch["s"][ix], batch["lab"][ix], batch["mask"][ix])
            a = batch["a"][ix]
            lp = lp_all.gather(1, a[:, None]).squeeze(1)
            ratio = torch.exp(lp - batch["logp"][ix])
            A = adv[ix]
            l1 = ratio * A
            l2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A
            pg = -torch.min(l1, l2).mean()
            p = lp_all.exp()
            ent = -(p * lp_all.masked_fill(~batch["mask"][ix], 0.0)).sum(-1).mean()
            lr_all = pol.logp_ref(batch["s"][ix], batch["mask"][ix])
            kl = (p * (lp_all - lr_all).masked_fill(~batch["mask"][ix], 0.0)).sum(-1).mean()
            loss = pg - args.ent * ent + klc * kl
            opt_p.zero_grad()
            loss.backward()
            if not train_alpha:
                pol.alpha.grad = None
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt_p.step()
            logit = critic(batch["phi"][ix])
            vloss = Fn.binary_cross_entropy_with_logits(logit, batch["tgt"][ix])
            opt_v.zero_grad()
            vloss.backward()
            opt_v.step()
            with torch.no_grad():
                stats["kl"] += kl.item()
                stats["entropy"] += ent.item()
                stats["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["approx_kl"] += (batch["logp"][ix] - lp).mean().item()
                stats["pg_loss"] += pg.item()
                stats["vloss"] += vloss.item()
            cnt += 1
    out = {k: v / max(1, cnt) for k, v in stats.items()}
    with torch.no_grad():
        v = torch.tanh(critic(batch["phi"]) / 2.0)
        r = batch["ret"]
        out["explained_var"] = float(1.0 - ((r - v).var() / (r.var() + 1e-8)))
    return out


# ----------------------------------------------------------------------------- main loops
def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def probe_opps(pool, specs, league_cost):
    """Keep only opponents that load and play a few turns."""
    ok = []
    w = default_weights(0)
    tasks = [(ro_worker.play_game, (0, 5, w, s, False, 0, False, 80)) for s in specs]
    res = [pool.apply_async(_call, (t,)) for t in tasks]
    for s, r in zip(specs, res):
        try:
            g = r.get(timeout=120)
            if g["err"]:
                print("skip opponent", s["name"], g["err"], flush=True)
            else:
                ok.append(s)
                league_cost[s["name"]] = g["wall"] / max(1, g["turns"])
        except Exception as e:
            print("skip opponent", s["name"], repr(e)[:100], flush=True)
    return ok


def _call(t):
    f, a = t
    return f(*a)


def train(args):
    run_dir = os.path.join(HERE, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=ro_worker.init_worker)
    torch.set_num_threads(max(1, min(args.workers, 3)))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    w0 = default_weights(args.hidden, seed=args.seed)
    if args.init:
        w0 = json.load(open(args.init))
    pol = ROPolicy(w0)
    critic = Critic()
    opt_p = torch.optim.Adam(pol.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr_v)
    it0 = 0
    st_path = os.path.join(run_dir, "state.pt")
    if args.resume and os.path.exists(st_path):
        st = torch.load(st_path)
        pol.load_state_dict(st["pol"]); critic.load_state_dict(st["critic"])
        opt_p.load_state_dict(st["opt_p"]); opt_v.load_state_dict(st["opt_v"])
        it0 = st["it"] + 1

    cost0 = {}
    specs = probe_opps(pool, resolve_opps([p for p in args.opps.split(",") if p]), cost0)
    league = League(specs, rng, snaps_keep=args.snaps_keep, p_self=args.p_self, cost_power=args.cost_power)
    league.cost.update(cost0)
    print("league:", [s["name"] for s in specs], flush=True)
    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    t_start = time.time()
    map_ctr = 0
    it = it0
    while True:
        elapsed_h = (time.time() - t_start) / 3600.0
        if (args.iters and it - it0 >= args.iters) or (args.hours and elapsed_h >= args.hours):
            break
        frac = max((it - it0) / args.iters if args.iters else 0.0,
                   elapsed_h / args.hours if args.hours else 0.0)
        beta = args.beta0 * max(0.0, 1.0 - frac / args.beta_anneal)
        klc = args.kl0 + (args.kl1 - args.kl0) * min(1.0, frac / args.beta_anneal)
        w = pol.to_json()
        if args.snap_every and it % args.snap_every == 0 and not args.no_self:
            league.add_snap("snap%d" % it, w)
        t0 = time.time()
        tasks = []
        for _ in range(max(1, args.games_per_iter // 2)):
            opp = league.sample()
            tasks.append({"map_idx": args.map_offset + map_ctr, "weights": w, "opp": opp,
                          "seed": rng.randrange(1 << 30), "sample": True})
            map_ctr += 1
        eps, per_opp, pair_means = [], collections.defaultdict(list), []
        turns_tot, ro_ms, nerr = 0, [], 0
        for pair in pool.imap_unordered(ro_worker.play_pair, tasks):
            outs = []
            for g in pair:
                if g["outcome"] is None:
                    continue
                if g["err"] or g.get("forfeit"):
                    nerr += 1       # bot crash / malformed move: not a real game result, excluded
                    continue
                eps.append((g["recs"], g["outcome"], g["turns"]))
                sc = 0.5 + 0.5 * g["outcome"]
                per_opp[g["opp"]].append(sc)
                league.win[g["opp"]].append(sc)
                outs.append(g["outcome"])
                turns_tot += g["turns"]
                league.note_cost(g["opp"], g["wall"], g["turns"])
                ro_ms.append(g["ro_ms"])
            if outs:
                pair_means.append(sum(outs) / len(outs))
        t_roll = time.time() - t0
        ndec = sum(len(e[0]) for e in eps)
        batch = build_batch(eps, critic, args.gamma, args.lam, beta, args.r_draw)
        t1 = time.time()
        st = {}
        if batch is not None:
            st = ppo_update(pol, critic, opt_p, opt_v, batch, args, klc, it >= args.alpha_freeze)
        wfile = os.path.join(run_dir, "weights_%d.json" % it)
        write_json_atomic(wfile, pol.to_json())
        write_json_atomic(os.path.join(run_dir, "weights_latest.json"), pol.to_json())
        torch.save({"pol": pol.state_dict(), "critic": critic.state_dict(), "opt_p": opt_p.state_dict(),
                    "opt_v": opt_v.state_dict(), "it": it}, st_path)
        rec = {"it": it, "games": len(eps), "decisions": ndec, "turns": turns_tot,
               "roll_s": round(t_roll, 1), "learn_s": round(time.time() - t1, 1),
               "games_per_s": round(len(eps) / max(1e-9, t_roll), 3),
               "dec_per_s": round(ndec / max(1e-9, t_roll), 1),
               "turns_per_s": round(turns_tot / max(1e-9, t_roll), 1),
               "ro_ms_per_turn": round(float(np.mean(ro_ms)) if ro_ms else 0, 2),
               "beta": round(beta, 4), "kl_coef": round(klc, 4), "errs": nerr,
               "pair_mean": round(float(np.mean(pair_means)) if pair_means else 0, 3),
               "winrate": round(float(np.mean([x for v in per_opp.values() for x in v])) if per_opp else 0, 3),
               "wr_by_opp": {k: [len(v), round(sum(v) / len(v), 3)] for k, v in sorted(per_opp.items())},
               "alpha": round(float(pol.alpha), 4),
               "rss_mb": round(_rss_mb(), 0)}
        rec.update({k: round(v, 5) for k, v in st.items()})
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps({k: rec[k] for k in ("it", "games", "decisions", "games_per_s", "dec_per_s",
                                              "winrate", "pair_mean") if k in rec}),
              {k: rec.get(k) for k in ("kl", "entropy", "clipfrac", "vloss", "explained_var")}, flush=True)
        it += 1
    pool.terminate()


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def evaluate(args):
    if args.eval_maps:
        os.environ["ARENA_MAPS"] = os.path.abspath(args.eval_maps)
    w = json.load(open(args.eval)) if args.eval != "ref" else default_weights(args.hidden)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=ro_worker.init_worker)
    opps = resolve_opps([p for p in args.eval_opp.split(",") if p])
    for opp in opps:
        tasks = [{"map_idx": args.map_offset + i, "weights": w, "opp": opp, "seed": 1000 + i,
                  "sample": args.eval_sample, "record": False} for i in range(args.eval_pairs)]
        t0 = time.time()
        sc, dec_ms, forf = [], [], 0
        for pair in pool.imap_unordered(ro_worker.play_pair, tasks):
            for g in pair:
                sc.append(0.5 + 0.5 * g["outcome"])
                dec_ms.append(g["ro_ms"])
                forf += 1 if g.get("err") else 0
        se = math.sqrt(np.var(sc) / max(1, len(sc)))
        print(json.dumps({"opp": opp["name"], "n": len(sc), "score": round(float(np.mean(sc)), 3),
                          "se": round(float(se), 3), "errs": forf,
                          "ms_per_turn": round(float(np.mean(dec_ms)), 2), "s": round(time.time() - t0)}),
              flush=True)
    pool.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games-per-iter", type=int, default=48)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--kl0", type=float, default=0.02)
    ap.add_argument("--kl1", type=float, default=0.005)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--gamma", type=float, default=0.998)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--r-draw", type=float, default=-0.1)
    ap.add_argument("--beta0", type=float, default=0.2)
    ap.add_argument("--beta-anneal", type=float, default=0.5, help="fraction of run at which beta (and KL) finish annealing")
    ap.add_argument("--alpha-freeze", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--init", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--map-offset", type=int, default=10000)
    ap.add_argument("--opps", default=",".join(DEFAULT_OPPS))
    ap.add_argument("--no-self", action="store_true")
    ap.add_argument("--snap-every", type=int, default=5)
    ap.add_argument("--snaps-keep", type=int, default=6)
    ap.add_argument("--cost-power", type=float, default=1.0, help="down-weight slow opponents by (ref/cost)^p")
    ap.add_argument("--p-self", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default="", help="weights json (or 'ref') to evaluate with argmax")
    ap.add_argument("--eval-opp", default="bots/versions/tune_base12g.py")
    ap.add_argument("--eval-pairs", type=int, default=40)
    ap.add_argument("--eval-maps", default="")
    ap.add_argument("--eval-sample", action="store_true")
    args = ap.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
